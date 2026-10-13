//! Immutable command JSON fragments with only the live position substituted.
//! This changes neither intent contents nor the pre-submit flush/freshness order.
use anyhow::{ensure, Result};
use serde_json::Value;
#[derive(Default)]
pub(super) struct CommandTemplate { pieces: Vec<Vec<u8>> }
impl CommandTemplate {
    pub fn new(commands: &[Value]) -> Result<Self> {
        let marker = "\"__w9100_live_position__\"";
        let mut pieces=Vec::new();
        let mut prefix=vec![b'['];
        for (i, command) in commands.iter().enumerate() {
            ensure!(command["context"].as_array().map(|x|x.len())==Some(3), "intent context schema");
            ensure!(command["context"][0].as_u64().is_some(), "intent position schema");
            let mut c=command.clone();c["context"][0]=Value::String("__w9100_live_position__".into());
            let bytes=serde_json::to_vec(&c)?;
            let spots:Vec<_>=bytes.windows(marker.len()).enumerate().filter(|(_,b)|*b==marker.as_bytes()).map(|(i,_)|i).collect();
            ensure!(spots.len()==1,"unique live position");let at=spots[0];
            if i>0 {prefix.push(b',');}
            prefix.extend_from_slice(&bytes[..at]);pieces.push(prefix);
            prefix=bytes[at+marker.len()..].to_vec();
        }
        prefix.push(b']');pieces.push(prefix);Ok(Self{pieces})
    }
    pub fn render(&self, position:usize)->Vec<u8> {
        let p=position.to_string();let mut bytes=Vec::with_capacity(self.pieces.iter().map(|x|x.len()+p.len()).sum());
        for (i,s) in self.pieces.iter().enumerate(){if i>0 {bytes.extend_from_slice(p.as_bytes());}bytes.extend_from_slice(s);}
        bytes
    }
}
#[cfg(test)]
mod tests {
 use super::*;
 #[test] fn cached_intent_all_positions_exact() {
  let commands=vec![serde_json::json!({"pipeline":"copy","context":[0,1,null],"bytes":8192,"predicted_ms":0.3,"src":"0xa","dst":"0xb"}),serde_json::json!({"pipeline":"q6k_routed","context":[0,1,22],"push":[2048,512,8,1,0,0,512],"grid":[128,4],"predicted_ms":12.0})];
  let t=CommandTemplate::new(&commands).unwrap();
  for position in 0..=4096 {let mut expected=commands.clone();for c in &mut expected {c["context"][0]=position.into();}let actual:Vec<Value>=serde_json::from_slice(&t.render(position)).unwrap();assert_eq!(actual,expected);}
  assert!(CommandTemplate::new(&[serde_json::json!({"context":[null,1,0]})]).is_err());
  assert_eq!(CommandTemplate::new(&[]).unwrap().render(0),b"[]");
 }
}
