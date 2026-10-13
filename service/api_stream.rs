// Additive host protocol only. Computational implementation is the pinned library.
use anyhow::{ensure, Result};
use lfm2_engine::{gpu_lfm::GpuEngine, model_file::ModelFile, text_output::DecodedText};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::time::Instant;
use w9100_tokenizer::{Tokenizer, ChatTemplate, EncodeOptions, DecodeOptions, StreamDecoder, StreamOptions, InvalidUtf8Policy};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request { id: String, text: Option<String>, chat: Option<Value>, tokens: Option<Vec<u32>>, n: usize }
fn emit(v: Value) -> Result<()> { let mut o=std::io::stdout().lock(); serde_json::to_writer(&mut o,&v)?; writeln!(o)?; o.flush()?; Ok(()) }
fn prepare(tok: &Tokenizer, line: &str) -> Result<(Request,Vec<u32>)> {
    let r: Request=serde_json::from_str(line)?;
    ensure!(!r.id.is_empty() && r.id.len()<=128,"invalid id");
    ensure!((1..=1024).contains(&r.n),"max_tokens must be 1..1024");
    ensure!(usize::from(r.text.is_some())+usize::from(r.chat.is_some())+usize::from(r.tokens.is_some())==1,"exactly one input required");
    let ids=if let Some(ids)=&r.tokens {ids.clone()} else if let Some(text)=&r.text {
        ensure!(!text.is_empty(),"empty prompt");
        tok.encode(text.as_bytes(),EncodeOptions{add_special:true,parse_special:false})?
    } else {
        let rendered=ChatTemplate::from_tokenizer(tok)?.render_json(r.chat.as_ref().unwrap())?;
        tok.encode(rendered.as_bytes(),EncodeOptions{add_special:false,parse_special:true})?
    };
    ensure!(!ids.is_empty() && ids.len()+r.n<=4096,"empty or context exceeds 4096");
    ensure!(ids.iter().all(|&x|(x as usize)<tok.vocab_size()),"out-of-range token");
    Ok((r,ids))
}
fn main() -> Result<()> {
    let args:Vec<String>=std::env::args().collect();
    ensure!(args.len()==3 && ["--model","--validate-only"].contains(&args[1].as_str()),"usage: api_stream --model|--validate-only MODEL");
    let tok=Tokenizer::from_gguf_file(&args[2])?;
    if args[1]=="--validate-only" {
        for line in std::io::stdin().lock().lines() {match prepare(&tok,&line?) {
            Ok((r,ids))=>emit(json!({"type":"validated","id":r.id,"prompt_tokens":ids}))?,
            Err(e)=>emit(json!({"type":"validation_error","message":e.to_string()}))?
        }} return Ok(());
    }
    let start=Instant::now();
    let mf=ModelFile::open(&args[2])?;
    let vram=std::path::Path::new("/sys/class/drm/card0/device");
    ensure!(std::fs::read_to_string(vram.join("device"))?.trim()=="0x67a0","wrong GPU");
    let total:u64=std::fs::read_to_string(vram.join("mem_info_vram_total"))?.trim().parse()?;
    let used:u64=std::fs::read_to_string(vram.join("mem_info_vram_used"))?.trim().parse()?;
    let weights:u64=mf.tensors.iter().map(|t| if std::env::var("LFM_Q6_ALIGNED").as_deref()==Ok("1") && t.ggml_type==14 {t.nbytes/210*212} else {t.nbytes}).sum();
    ensure!(total.saturating_sub(used)>weights+768*1024*1024,"insufficient co-resident VRAM headroom");
    let mut engine=GpuEngine::load(&mf)?; engine.set_ctx(4096);
    emit(json!({"type":"ready","model_load_count":1,"pid":std::process::id(),"load_ms":start.elapsed().as_secs_f64()*1000.0,"vram_before":used,"weight_bytes":weights}))?;
    let mut count=0;
    for line in std::io::stdin().lock().lines() {
        let line=line?;
        let (r,prompt)=match prepare(&tok,&line) {Ok(x)=>x,Err(e)=>{emit(json!({"type":"validation_error","message":e.to_string(),"gpu_dispatches":engine.gpu.dispatch_count()}))?;continue;}};
        emit(json!({"type":"accepted","id":r.id,"prompt_tokens":prompt,"gpu_dispatches":engine.gpu.dispatch_count()}))?;
        let start=Instant::now();
        engine.reset_session()?;
        let mut pos=0;let mut next=0;
        engine.gpu.ctx.set_submission_context("prefill",prompt.len());
        for chunk in prompt.chunks(lfm2_engine::work_bound::PREFILL_CHUNK) {
            let t=lfm2_engine::vulkan::profile_clock_ns();
            next=engine.forward_token(chunk,pos)?;
            engine.gpu.ctx.profile_forward(t,lfm2_engine::vulkan::profile_clock_ns(),pos,chunk.len())?;
            pos+=chunk.len();
        }
        let prefill_ms=start.elapsed().as_secs_f64()*1000.0;
        let mut decoder=StreamDecoder::new(&tok,StreamOptions{render_special:false,invalid:InvalidUtf8Policy::Replace});
        let mut cleanup=DecodedText::new(tok.clean_spaces());let mut streamed=String::new();let mut out=Vec::new();
        for _ in 0..r.n {
            let token=next;if token==engine.hp.eos {break;}
            out.push(token);
            let delta=cleanup.push(&decoder.push_token(token)?);streamed.push_str(&delta);
            emit(json!({"type":"delta","id":r.id,"text":delta,"token":token}))?;
            engine.gpu.ctx.set_submission_context("decode",prompt.len());
            let t=lfm2_engine::vulkan::profile_clock_ns();
            next=engine.forward_token(&[token],pos)?;
            engine.gpu.ctx.profile_forward(t,lfm2_engine::vulkan::profile_clock_ns(),pos,1)?;
            pos+=1;
        }
        let delta=cleanup.finish(&decoder.finish()?);streamed.push_str(&delta);
        if !delta.is_empty(){emit(json!({"type":"delta","id":r.id,"text":delta}))?;}
        let decoded=tok.decode(&out,DecodeOptions{remove_special:true,render_special:false})?;
        ensure!(String::from_utf8_lossy(&decoded)==streamed,"stream decode mismatch");
        count+=1;
        emit(json!({"type":"result","id":r.id,"prompt_tokens":prompt,"tokens":out,"text":streamed,"finish_reason":if out.len()==r.n {"length"}else{"stop"},"request_count":count,"model_load_count":1,"pid":std::process::id(),"prefill_ms":prefill_ms,"request_ms":start.elapsed().as_secs_f64()*1000.0,"gpu_dispatches":engine.gpu.dispatch_count()}))?;
    } Ok(())
}
