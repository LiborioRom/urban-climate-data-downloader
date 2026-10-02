from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=False)

from agent.dialogue_manager import interpret_turn
from agent.local_llm import ollama_status
from schemas.conversation import ConversationState, Intent


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprueba Ollama y el modelo local configurado.")
    parser.add_argument("--inference", action="store_true", help="Ejecuta además una inferencia estructurada corta.")
    args = parser.parse_args()

    status = ollama_status(timeout=5)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not status["running"]:
        print("ERROR: Ollama no responde. Abre Ollama o ejecuta `ollama serve`.", file=sys.stderr)
        return 1
    if not status["model_available"]:
        print(f"ERROR: falta {status['model']}. Ejecuta `ollama pull {status['model']}`.", file=sys.stderr)
        return 2
    if args.inference:
        for _ in range(2):
            turn, provider = interpret_turn(
                "Quiero LST y NDVI de mayo a octubre de 2024, cada 100 metros.",
                ConversationState(),
                [],
            )
            if provider == "ollama" and turn.intent == Intent.CREATE_OR_UPDATE_REQUEST and turn.request_patch:
                break
        else:
            print(f"Última respuesta ({provider}): {turn.model_dump_json()}", file=sys.stderr)
            print("ERROR: el modelo respondió, pero no interpretó correctamente la petición de prueba.", file=sys.stderr)
            return 3
        print(turn.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
