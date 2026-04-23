use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::Client as HttpClient;
use serde_json::{Map, Value};

pub const DEFAULT_MODEL: &str = "gpt-5.4-mini";
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Clone, Debug)]
pub struct Client {
    http: HttpClient,
    api_key: String,
    base_url: String,
}

impl Client {
    pub fn init(api_key: impl Into<String>) -> Result<Self> {
        let http = HttpClient::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(300))
            .user_agent(format!("agentz/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self {
            http,
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
        })
    }

    pub fn from_env() -> Result<Self> {
        let env_api_key = normalize_env_value(env::var("OPENAI_API_KEY").ok());
        let env_base_url = normalize_env_value(env::var("OPENAI_BASE_URL").ok());

        let dotenv_body = if env_api_key.is_none() || env_base_url.is_none() {
            read_dotenv_local().ok()
        } else {
            None
        };

        let config = resolve_client_config(env_api_key, env_base_url, dotenv_body.as_deref())?;
        Ok(Self::init(config.api_key)?.with_base_url(config.base_url))
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = normalize_base_url(base_url.into());
        self
    }

    pub fn create_response(&self, request: ResponsesRequest) -> Result<Response> {
        if request.stream.unwrap_or(false) {
            return self.create_response_streaming(request, |_| Ok(()), |_| Ok(()));
        }

        let payload = request.to_json_value()?;
        let response = self
            .http
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .header("accept-encoding", "identity")
            .json(&payload)
            .send()
            .context("failed to send OpenAI request")?;

        let status = response.status();
        let raw_body = response
            .text()
            .context("failed to read OpenAI response body")?;
        if !status.is_success() {
            bail!("OpenAI request failed: status={} body={}", status, raw_body);
        }

        Response::from_body(raw_body)
    }

    pub fn create_response_streaming<F, G>(
        &self,
        request: ResponsesRequest,
        mut on_output_text_delta: F,
        mut on_response_completed: G,
    ) -> Result<Response>
    where
        F: FnMut(&str) -> Result<()>,
        G: FnMut(&Response) -> Result<()>,
    {
        let mut request = request;
        request.stream = Some(true);

        let payload = request.to_json_value()?;
        let response = self
            .http
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .header("accept-encoding", "identity")
            .header("accept", "text/event-stream")
            .json(&payload)
            .send()
            .context("failed to send streaming OpenAI request")?;

        let status = response.status();
        if !status.is_success() {
            let raw_body = response
                .text()
                .context("failed to read failed OpenAI stream body")?;
            bail!("OpenAI request failed: status={} body={}", status, raw_body);
        }

        read_streamed_response_from_reader(
            BufReader::new(response),
            &mut on_output_text_delta,
            &mut on_response_completed,
        )
    }
}

#[derive(Clone, Debug)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Input,
    pub instructions: Option<String>,
    pub tools: Vec<Tool>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub previous_response_id: Option<String>,
    pub store: Option<bool>,
    pub stream: Option<bool>,
}

impl ResponsesRequest {
    pub fn init(model: impl Into<String>, input: Input) -> Self {
        Self {
            model: model.into(),
            input,
            instructions: None,
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            store: None,
            stream: None,
        }
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }

    pub fn with_store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn followup(
        &self,
        previous_response_id: impl Into<String>,
        input_items: Vec<InputItem>,
    ) -> Self {
        Self {
            model: self.model.clone(),
            input: Input::items(input_items),
            instructions: self.instructions.clone(),
            tools: self.tools.clone(),
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            previous_response_id: Some(previous_response_id.into()),
            store: self.store,
            stream: self.stream,
        }
    }

    pub fn to_json_value(&self) -> Result<Value> {
        let mut object = Map::new();
        object.insert("model".to_owned(), Value::String(self.model.clone()));
        object.insert("input".to_owned(), self.input.to_json_value()?);

        if let Some(instructions) = &self.instructions {
            object.insert(
                "instructions".to_owned(),
                Value::String(instructions.clone()),
            );
        }
        if !self.tools.is_empty() {
            object.insert(
                "tools".to_owned(),
                Value::Array(
                    self.tools
                        .iter()
                        .map(Tool::to_json_value)
                        .collect::<Result<Vec<_>>>()?,
                ),
            );
        }
        if let Some(tool_choice) = self.tool_choice {
            object.insert(
                "tool_choice".to_owned(),
                Value::String(tool_choice.as_str().to_owned()),
            );
        }
        if let Some(parallel_tool_calls) = self.parallel_tool_calls {
            object.insert(
                "parallel_tool_calls".to_owned(),
                Value::Bool(parallel_tool_calls),
            );
        }
        if let Some(previous_response_id) = &self.previous_response_id {
            object.insert(
                "previous_response_id".to_owned(),
                Value::String(previous_response_id.clone()),
            );
        }
        if let Some(store) = self.store {
            object.insert("store".to_owned(), Value::Bool(store));
        }
        if let Some(stream) = self.stream {
            object.insert("stream".to_owned(), Value::Bool(stream));
        }

        Ok(Value::Object(object))
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(self.to_json_value()?.to_string())
    }
}

#[derive(Clone, Debug)]
pub enum Input {
    Text(String),
    Items(Vec<InputItem>),
}

impl Input {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    pub fn items(items: Vec<InputItem>) -> Self {
        Self::Items(items)
    }

    fn to_json_value(&self) -> Result<Value> {
        match self {
            Self::Text(text) => Ok(Value::String(text.clone())),
            Self::Items(items) => Ok(Value::Array(
                items
                    .iter()
                    .map(InputItem::to_json_value)
                    .collect::<Result<Vec<_>>>()?,
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub enum InputItem {
    Message(Message),
    MessageText { role: Role, text: String },
    FunctionCallOutput(FunctionCallOutput),
    RawJson(Value),
}

impl InputItem {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::MessageText {
            role: Role::User,
            text: text.into(),
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::MessageText {
            role: Role::Assistant,
            text: text.into(),
        }
    }

    pub fn function_call_output(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self::FunctionCallOutput(FunctionCallOutput {
            call_id: call_id.into(),
            output: output.into(),
        })
    }

    pub fn raw_json_str(json: &str) -> Result<Self> {
        Ok(Self::RawJson(
            serde_json::from_str(json).context("invalid raw JSON input item")?,
        ))
    }

    pub fn raw_json_value(value: Value) -> Self {
        Self::RawJson(value)
    }

    fn to_json_value(&self) -> Result<Value> {
        match self {
            Self::Message(message) => {
                let mut object = Map::new();
                object.insert("type".to_owned(), Value::String("message".to_owned()));
                object.insert(
                    "role".to_owned(),
                    Value::String(message.role.as_str().to_owned()),
                );
                object.insert(
                    "content".to_owned(),
                    Value::Array(
                        message
                            .content
                            .iter()
                            .map(MessageInputContent::to_json_value)
                            .collect::<Vec<_>>(),
                    ),
                );
                Ok(Value::Object(object))
            }
            Self::MessageText { role, text } => {
                let mut object = Map::new();
                object.insert("type".to_owned(), Value::String("message".to_owned()));
                object.insert("role".to_owned(), Value::String(role.as_str().to_owned()));
                object.insert(
                    "content".to_owned(),
                    Value::Array(vec![
                        MessageInputContent::input_text(text.clone()).to_json_value(),
                    ]),
                );
                Ok(Value::Object(object))
            }
            Self::FunctionCallOutput(output) => {
                let mut object = Map::new();
                object.insert(
                    "type".to_owned(),
                    Value::String("function_call_output".to_owned()),
                );
                object.insert("call_id".to_owned(), Value::String(output.call_id.clone()));
                object.insert("output".to_owned(), Value::String(output.output.clone()));
                Ok(Value::Object(object))
            }
            Self::RawJson(value) => Ok(value.clone()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: Vec<MessageInputContent>,
}

#[derive(Clone, Debug)]
pub struct FunctionCallOutput {
    pub call_id: String,
    pub output: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }
}

#[derive(Clone, Debug)]
pub enum MessageInputContent {
    InputText(String),
}

impl MessageInputContent {
    pub fn input_text(text: impl Into<String>) -> Self {
        Self::InputText(text.into())
    }

    fn to_json_value(&self) -> Value {
        match self {
            Self::InputText(text) => {
                let mut object = Map::new();
                object.insert("type".to_owned(), Value::String("input_text".to_owned()));
                object.insert("text".to_owned(), Value::String(text.clone()));
                Value::Object(object)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub parameters_json: String,
    pub description: Option<String>,
    pub strict: Option<bool>,
}

impl Tool {
    pub fn function(name: impl Into<String>, parameters_json: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters_json: parameters_json.into(),
            description: None,
            strict: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    fn to_json_value(&self) -> Result<Value> {
        let parameters = serde_json::from_str(&self.parameters_json)
            .with_context(|| format!("invalid tool schema for {}", self.name))?;

        let mut object = Map::new();
        object.insert("type".to_owned(), Value::String("function".to_owned()));
        object.insert("name".to_owned(), Value::String(self.name.clone()));
        if let Some(description) = &self.description {
            object.insert("description".to_owned(), Value::String(description.clone()));
        }
        object.insert("parameters".to_owned(), parameters);
        if let Some(strict) = self.strict {
            object.insert("strict".to_owned(), Value::Bool(strict));
        }
        Ok(Value::Object(object))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    Required,
    None,
}

impl ToolChoice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Required => "required",
            Self::None => "none",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonBlob {
    pub bytes: String,
}

#[derive(Clone, Debug)]
pub struct Response {
    body: String,
    parsed: Value,
}

impl Response {
    pub fn from_body(body: String) -> Result<Self> {
        let parsed = serde_json::from_str(&body).context("failed to parse OpenAI response JSON")?;
        Ok(Self { body, parsed })
    }

    pub fn body(&self) -> &str {
        &self.body
    }

    pub fn id(&self) -> Result<&str> {
        string_field(self.root_object()?, "id")
    }

    pub fn output_text(&self) -> Result<String> {
        let mut out = String::new();

        for item in self.output_array()? {
            let item_object = match item.as_object() {
                Some(object) => object,
                None => continue,
            };
            if string_field(item_object, "type").ok() != Some("message") {
                continue;
            }

            let Some(content) = item_object.get("content").and_then(Value::as_array) else {
                continue;
            };

            for content_item in content {
                let Some(content_object) = content_item.as_object() else {
                    continue;
                };
                if string_field(content_object, "type").ok() != Some("output_text") {
                    continue;
                }
                out.push_str(string_field(content_object, "text")?);
            }
        }

        Ok(out)
    }

    pub fn output_items(&self) -> Result<Vec<Value>> {
        Ok(self.output_array()?.clone())
    }

    pub fn output_item_json_blobs(&self) -> Result<Vec<JsonBlob>> {
        self.output_array()?
            .iter()
            .map(|item| {
                Ok(JsonBlob {
                    bytes: serde_json::to_string(item)
                        .context("failed to reserialize output item")?,
                })
            })
            .collect()
    }

    pub fn function_calls(&self) -> Result<Vec<FunctionCall>> {
        let mut calls = Vec::new();

        for item in self.output_array()? {
            let Some(item_object) = item.as_object() else {
                continue;
            };
            if string_field(item_object, "type").ok() != Some("function_call") {
                continue;
            }

            calls.push(FunctionCall {
                id: optional_string_field(item_object, "id").map(ToOwned::to_owned),
                call_id: string_field(item_object, "call_id")?.to_owned(),
                name: string_field(item_object, "name")?.to_owned(),
                arguments: string_field(item_object, "arguments")?.to_owned(),
            });
        }

        Ok(calls)
    }

    fn root_object(&self) -> Result<&Map<String, Value>> {
        self.parsed
            .as_object()
            .ok_or_else(|| anyhow!("invalid OpenAI response"))
    }

    fn output_array(&self) -> Result<&Vec<Value>> {
        self.root_object()?
            .get("output")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("invalid OpenAI response"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionCall {
    pub id: Option<String>,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

impl FunctionCall {
    pub fn arguments_json(&self) -> Result<Value> {
        serde_json::from_str(&self.arguments).context("failed to parse function call arguments")
    }
}

pub fn run_agent_loop<F>(
    client: &Client,
    initial_request: ResponsesRequest,
    max_turns: usize,
    mut handler: F,
) -> Result<Response>
where
    F: FnMut(&FunctionCall) -> Result<String>,
{
    let mut request = initial_request;

    for _ in 0..max_turns {
        let response = client.create_response(request.clone())?;
        let calls = response.function_calls()?;
        if calls.is_empty() {
            return Ok(response);
        }

        let previous_response_id = response.id()?.to_owned();
        let outputs = calls
            .iter()
            .map(|call| {
                Ok(InputItem::function_call_output(
                    call.call_id.clone(),
                    handler(call)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        request = request.followup(previous_response_id, outputs);
    }

    bail!("agent loop exceeded max turns")
}

pub fn read_streamed_response_from_reader<R, F, G>(
    reader: R,
    on_output_text_delta: &mut F,
    on_response_completed: &mut G,
) -> Result<Response>
where
    R: Read,
    F: FnMut(&str) -> Result<()>,
    G: FnMut(&Response) -> Result<()>,
{
    let mut reader = BufReader::new(reader);
    let mut event_name = String::new();
    let mut data = String::new();
    let mut line = String::new();

    loop {
        line.clear();
        let read_n = reader
            .read_line(&mut line)
            .context("failed to read SSE line")?;
        if read_n == 0 {
            break;
        }

        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            if let Some(response) = dispatch_sse_event(
                &event_name,
                &data,
                on_output_text_delta,
                on_response_completed,
            )? {
                return Ok(response);
            }
            event_name.clear();
            data.clear();
            continue;
        }

        if trimmed.starts_with(':') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("event:") {
            event_name.clear();
            event_name.push_str(sse_field_value(rest));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(sse_field_value(rest));
        }
    }

    if let Some(response) = dispatch_sse_event(
        &event_name,
        &data,
        on_output_text_delta,
        on_response_completed,
    )? {
        return Ok(response);
    }

    bail!("OpenAI stream ended without completion")
}

fn dispatch_sse_event<F, G>(
    event_name: &str,
    data: &str,
    on_output_text_delta: &mut F,
    on_response_completed: &mut G,
) -> Result<Option<Response>>
where
    F: FnMut(&str) -> Result<()>,
    G: FnMut(&Response) -> Result<()>,
{
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }

    let parsed: Value = serde_json::from_str(data).context("failed to parse SSE event JSON")?;
    let object = parsed
        .as_object()
        .ok_or_else(|| anyhow!("invalid SSE event payload"))?;

    let event_type = if event_name.is_empty() {
        string_field(object, "type")?
    } else {
        event_name
    };

    match event_type {
        "response.output_text.delta" => {
            on_output_text_delta(string_field(object, "delta")?)?;
            Ok(None)
        }
        "response.completed" => {
            let response_value = object
                .get("response")
                .ok_or_else(|| anyhow!("invalid OpenAI response"))?;
            let response = response_from_value(response_value)?;
            on_response_completed(&response)?;
            Ok(Some(response))
        }
        "response.failed" => bail!("OpenAI request failed"),
        "response.incomplete" => bail!("OpenAI response incomplete"),
        _ => Ok(None),
    }
}

fn response_from_value(value: &Value) -> Result<Response> {
    Response::from_body(
        serde_json::to_string(value).context("failed to serialize completed response payload")?,
    )
}

fn string_field<'a>(object: &'a Map<String, Value>, field_name: &str) -> Result<&'a str> {
    object
        .get(field_name)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("invalid OpenAI response"))
}

fn optional_string_field<'a>(object: &'a Map<String, Value>, field_name: &str) -> Option<&'a str> {
    object.get(field_name).and_then(Value::as_str)
}

fn normalize_base_url(base_url: String) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        DEFAULT_BASE_URL.to_owned()
    } else {
        trimmed.to_owned()
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ClientConfig {
    api_key: String,
    base_url: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct DotEnvValues {
    api_key: Option<String>,
    base_url: Option<String>,
}

fn resolve_client_config(
    env_api_key: Option<String>,
    env_base_url: Option<String>,
    body: Option<&str>,
) -> Result<ClientConfig> {
    let file_values = if env_api_key.is_none() || env_base_url.is_none() {
        body.map(parse_dotenv).transpose()?.unwrap_or_default()
    } else {
        DotEnvValues::default()
    };

    Ok(ClientConfig {
        api_key: env_api_key
            .or(file_values.api_key)
            .ok_or_else(|| anyhow!("missing OPENAI_API_KEY"))?,
        base_url: env_base_url
            .or(file_values.base_url)
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
    })
}

fn normalize_env_value(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn read_dotenv_local() -> Result<String> {
    fs::read_to_string(Path::new(".env.local")).context("failed to read .env.local")
}

fn parse_dotenv(body: &str) -> Result<DotEnvValues> {
    let mut values = DotEnvValues::default();

    for raw_line in body.lines() {
        let line = raw_line.trim_end_matches('\r');
        let Some((key, value)) = parse_dotenv_line(line)? else {
            continue;
        };

        match key.as_str() {
            "OPENAI_API_KEY" => values.api_key = Some(value),
            "OPENAI_BASE_URL" => values.base_url = Some(value),
            _ => {}
        }
    }

    Ok(values)
}

fn parse_dotenv_line(raw_line: &str) -> Result<Option<(String, String)>> {
    let mut line = raw_line.trim();
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }

    if let Some(rest) = line.strip_prefix("export") {
        if rest.is_empty() || rest.starts_with(char::is_whitespace) {
            line = rest.trim_start();
        }
    }

    let Some(eq_index) = line.find('=') else {
        return Ok(None);
    };

    let key = line[..eq_index].trim();
    if key.is_empty() {
        return Ok(None);
    }

    Ok(Some((
        key.to_owned(),
        parse_dotenv_value(&line[eq_index + 1..])?,
    )))
}

fn parse_dotenv_value(raw_value: &str) -> Result<String> {
    let value = raw_value.trim();
    if value.is_empty() {
        return Ok(String::new());
    }

    let quote = value.as_bytes()[0];
    if quote == b'"' || quote == b'\'' {
        let quote = quote as char;
        let mut end = None;
        for (idx, ch) in value.char_indices().skip(1) {
            if ch == quote {
                end = Some(idx);
                break;
            }
        }

        let Some(end) = end else {
            bail!("invalid dotenv");
        };

        let trailing = value[end + quote.len_utf8()..].trim();
        if !trailing.is_empty() && !trailing.starts_with('#') {
            bail!("invalid dotenv");
        }
        return Ok(value[1..end].to_owned());
    }

    Ok(strip_inline_comment(value).trim_end().to_owned())
}

fn strip_inline_comment(value: &str) -> &str {
    for (idx, ch) in value.char_indices() {
        if ch == '#' {
            let prev = value[..idx].chars().next_back();
            if prev.is_none() || prev.is_some_and(char::is_whitespace) {
                return &value[..idx];
            }
        }
    }
    value
}

fn sse_field_value(value: &str) -> &str {
    value.strip_prefix(' ').unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn serializes_flat_function_tool_shape() {
        let request = ResponsesRequest::init("gpt-5", Input::text("hello"))
            .with_tools(vec![
                Tool::function(
                    "get_weather",
                    r#"{
  "type": "object",
  "properties": {
    "location": { "type": "string" },
    "units": { "type": ["string", "null"], "enum": ["celsius", "fahrenheit"] }
  },
  "required": ["location", "units"],
  "additionalProperties": false
}"#,
                )
                .with_description("Look up weather")
                .with_strict(true),
            ])
            .with_parallel_tool_calls(false)
            .with_store(true);

        let actual = request.to_json_value().unwrap();
        let expected: Value = serde_json::from_str(
            r#"{
                "model":"gpt-5",
                "input":"hello",
                "tools":[{
                    "type":"function",
                    "name":"get_weather",
                    "description":"Look up weather",
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "location":{"type":"string"},
                            "units":{"type":["string","null"],"enum":["celsius","fahrenheit"]}
                        },
                        "required":["location","units"],
                        "additionalProperties":false
                    },
                    "strict":true
                }],
                "parallel_tool_calls":false,
                "store":true
            }"#,
        )
        .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn serializes_stream_flag_and_raw_input_items() {
        let request = ResponsesRequest::init(
            "gpt-5",
            Input::items(vec![
                InputItem::raw_json_str(r#"{"type":"reasoning","id":"rs_123"}"#).unwrap(),
                InputItem::user_text("hi"),
            ]),
        )
        .with_stream(true);

        let actual = request.to_json_value().unwrap();
        let expected: Value = serde_json::from_str(
            r#"{
                "model":"gpt-5",
                "input":[
                    {"type":"reasoning","id":"rs_123"},
                    {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}
                ],
                "stream":true
            }"#,
        )
        .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn followup_uses_function_call_output_items() {
        let first = ResponsesRequest::init("gpt-5", Input::items(vec![InputItem::user_text("hi")]))
            .with_instructions("be concise");

        let second = first.followup(
            "resp_123",
            vec![InputItem::function_call_output(
                "call_123",
                r#"{"ok":true}"#,
            )],
        );

        let actual = second.to_json_value().unwrap();
        let expected: Value = serde_json::from_str(
            r#"{
                "model":"gpt-5",
                "input":[{"type":"function_call_output","call_id":"call_123","output":"{\"ok\":true}"}],
                "instructions":"be concise",
                "previous_response_id":"resp_123"
            }"#,
        )
        .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn dotenv_parsing_and_resolution_prefer_live_env_values() {
        let dotenv = r#"# comment
export OPENAI_API_KEY="sk-file"
OPENAI_BASE_URL='https://dotenv.example/v1/' # trailing comment
"#;

        let parsed = parse_dotenv(dotenv).unwrap();
        assert_eq!(parsed.api_key.as_deref(), Some("sk-file"));
        assert_eq!(
            parsed.base_url.as_deref(),
            Some("https://dotenv.example/v1/")
        );

        let config =
            resolve_client_config(Some("  sk-env  ".trim().to_owned()), None, Some(dotenv))
                .unwrap();
        assert_eq!(config.api_key, "sk-env");
        assert_eq!(config.base_url, "https://dotenv.example/v1/");
    }

    #[test]
    fn dotenv_resolution_prefers_environment_over_dotenv() {
        let config = resolve_client_config(
            Some("  sk-env  ".trim().to_owned()),
            None,
            Some(
                r#"export OPENAI_API_KEY="sk-file"
OPENAI_BASE_URL=https://dotenv.example/v1/ # trailing comment"#,
            ),
        )
        .unwrap();

        assert_eq!(config.api_key, "sk-env");
        assert_eq!(config.base_url, "https://dotenv.example/v1/");
    }

    #[test]
    fn response_output_text_joins_output_text_segments_and_extracts_function_calls() {
        let response = Response::from_body(
            r#"{
                "id":"resp_123",
                "output":[
                    {"type":"reasoning"},
                    {
                        "type":"message",
                        "role":"assistant",
                        "content":[
                            {"type":"output_text","text":"hello"},
                            {"type":"output_text","text":" world"}
                        ]
                    },
                    {
                        "type":"function_call",
                        "call_id":"call_123",
                        "name":"noop",
                        "arguments":"{}"
                    }
                ]
            }"#
            .to_owned(),
        )
        .unwrap();

        assert_eq!(response.output_text().unwrap(), "hello world");

        let blobs = response.output_item_json_blobs().unwrap();
        assert_eq!(blobs.len(), 3);
        assert_eq!(blobs[0].bytes, r#"{"type":"reasoning"}"#);

        let calls = response.function_calls().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "noop");
    }

    #[test]
    fn stream_parser_emits_deltas_and_returns_completed_response() {
        let input = Cursor::new(
            "event: response.output_text.delta\n\
             data: {\"type\":\"response.output_text.delta\",\"delta\":\"hel\"}\n\
             \n\
             event: response.output_text.delta\n\
             data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}\n\
             \n\
             event: response.completed\n\
             data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_123\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\"}]}]}}\n\
             \n",
        );

        let mut deltas = String::new();
        let mut completed = false;
        let mut completed_id = None::<String>;

        let response = read_streamed_response_from_reader(
            input,
            &mut |delta| {
                deltas.push_str(delta);
                Ok(())
            },
            &mut |response| {
                completed = true;
                completed_id = Some(response.id()?.to_owned());
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(deltas, "hello");
        assert!(completed);
        assert_eq!(completed_id.as_deref(), Some("resp_123"));
        assert_eq!(response.output_text().unwrap(), "hello");
    }
}
