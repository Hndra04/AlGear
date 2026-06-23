"""CLI entry point for running the inference API server."""

import typer

app = typer.Typer()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to listen on"),
):
    """Start the PPE Compliance Inference API server."""
    from algear.api.app import run_server

    run_server(host=host, port=port)


if __name__ == "__main__":
    app()
