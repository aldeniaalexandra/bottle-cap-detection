"""
Main entry point for the bsort CLI.
"""
import typer
import yaml
from bsort.training.trainer import train_model
from bsort.inference.infer import run_inference

app = typer.Typer()

@app.command()
def train(config: str = typer.Option(..., help="Path to config yaml file")):
    """
    Train the model using the provided configuration.
    """
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    train_model(cfg)

@app.command()
def infer(config: str = typer.Option(..., help="Path to config yaml file")):
    """
    Run inference using the provided configuration.
    """
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    run_inference(cfg)

if __name__ == "__main__":
    app()
