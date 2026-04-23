use std::env;
use std::io::{self, Write};

use agentz::openai::{self, Input, ResponsesRequest};
use anyhow::Result;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    match args.next() {
        Some(arg) if arg == "-h" || arg == "--help" => print_usage(),
        Some(arg) if arg == "once" => {
            let prompt = join_args(args)
                .unwrap_or_else(|| "Reply with exactly: API wiring works.".to_owned());
            run_once(&prompt)
        }
        Some(arg) if arg == "tui" => {
            let initial_prompt = join_args(args);
            agentz::tui::run(agentz::tui::Options { initial_prompt })
        }
        Some(arg) => {
            let initial_prompt = join_args(std::iter::once(arg).chain(args));
            agentz::tui::run(agentz::tui::Options { initial_prompt })
        }
        None => agentz::tui::run(agentz::tui::Options::default()),
    }
}

fn run_once(prompt: &str) -> Result<()> {
    let client = openai::Client::from_env()?;
    let response = client.create_response(ResponsesRequest::init(
        openai::DEFAULT_MODEL,
        Input::text(prompt),
    ))?;

    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{}", response.output_text()?)?;
    Ok(())
}

fn join_args<I>(args: I) -> Option<String>
where
    I: IntoIterator<Item = String>,
{
    let joined = args.into_iter().collect::<Vec<_>>().join(" ");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

fn print_usage() -> Result<()> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(
        b"agentz\n  Launch the local TUI prototype.\n\nagentz [initial prompt words...]\n  Launch the TUI with the composer pre-filled.\n\nagentz once [prompt]\n  Run the old one-shot OpenAI Responses request.\n\n",
    )?;
    Ok(())
}
