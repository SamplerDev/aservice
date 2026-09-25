import os
import io
import time
import base64
import logging
import json
from enum import Enum
from typing import Optional, List, Dict, Any
from PIL import Image
from fastapi import FastAPI, HTTPException, Security, Depends, status, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
import asyncio

MODELS_FALLBACK = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

# ------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING ESTRUCTURADO
# ------------------------------------------------------------------
class DefaultTagFilter(logging.Filter):
    """Inyecta un tag 'SYSTEM' por defecto para logs de librerías externas (google.genai, uvicorn, etc.)"""
    def filter(self, record):
        if not hasattr(record, "tag"):
            record.tag = "SYSTEM"
        return True

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(tag)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Aplicar el filtro al handler principal
for handler in logging.getLogger().handlers:
    handler.addFilter(DefaultTagFilter())

class Logger:
    @staticmethod
    def info(tag: str, msg: str, data: Any = None):
        extra = f"\n{json.dumps(data, indent=2, ensure_ascii=False)}" if isinstance(data, (dict, list)) else (f"\n{data}" if data else "")
        logging.info(f"{msg}{extra}", extra={"tag": tag})

    @staticmethod
    def success(tag: str, msg: str, data: Any = None):
        extra = f"\n{json.dumps(data, indent=2, ensure_ascii=False)}" if isinstance(data, (dict, list)) else (f"\n{data}" if data else "")
        logging.info(f"✅ {msg}{extra}", extra={"tag": tag})

    @staticmethod
    def warn(tag: str, msg: str, data: Any = None):
        extra = f"\n{json.dumps(data, indent=2, ensure_ascii=False)}" if isinstance(data, (dict, list)) else (f"\n{data}" if data else "")
        logging.warning(f"⚠️ {msg}{extra}", extra={"tag": tag})

    @staticmethod
    def error(tag: str, msg: str, data: Any = None):
        extra = f"\n{json.dumps(data, indent=2, ensure_ascii=False)}" if isinstance(data, (dict, list)) else (f"\n{data}" if data else "")
        logging.error(f"❌ {msg}{extra}", extra={"tag": tag})

log = Logger()
# ------------------------------------------------------------------
# CARGA DE ENTORNO E INICIALIZACIÓN
# ------------------------------------------------------------------
load_dotenv()

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

log.info("STARTUP", "Iniciando AI Agent Microservice (Gemini 2.5 Flash)...")
log.info("ENV_CHECK", "Estado de variables de entorno:", {
    "INTERNAL_API_KEY": "✅ Configurada" if INTERNAL_API_KEY else "❌ FALTANTE",
    "GEMINI_API_KEY": "✅ Configurada" if GEMINI_API_KEY else "❌ FALTANTE"
})

if not GEMINI_API_KEY:
    log.error("GEMINI", "Error crítico: GEMINI_API_KEY no se encuentra definida.")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="AI Agentic Microservice (Gemini)")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# ------------------------------------------------------------------
# MIDDLEWARE GLOBAL DE LOGGING Y SEGURIDAD
# ------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_host = request.client.host if request.client else "Unknown"
    log.info("HTTP_IN", f"--> {request.method} {request.url.path} [IP: {client_host}]")
    
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    
    log.info("HTTP_OUT", f"<-- {request.method} {request.url.path} {response.status_code} [{duration}ms]")
    return response

async def verify_internal_key(api_key: str = Security(api_key_header)):
    if not INTERNAL_API_KEY or api_key != INTERNAL_API_KEY:
        log.error("AUTH_FAIL", "Intento de acceso rechazado: X-API-Key inválida o ausente.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Acceso no autorizado: X-API-Key inválida o ausente"
        )
    log.success("AUTH_OK", "Autenticación interna validada correctamente.")

# ------------------------------------------------------------------
# ESQUEMAS PYDANTIC (Structured Outputs)
# ------------------------------------------------------------------




class OfertaViaje(BaseModel):
    destino: str = Field(description="Ciudad, región o país principal del viaje")
    fecha_salida: str = Field(description="Fecha en formato YYYY-MM-DD o aproximada/mes")
    descripcion: str = Field(description="Resumen de inclusiones, estadía, transporte y precio")
    contacto: Optional[str] = Field(None, description="Teléfono o medio de contacto extraído")
    cupos: Optional[int] = Field(1, description="Cantidad estimada de cupos o lugares disponibles mencionados")

# NUEVO: Permite extraer 1 o N ofertas de una sola imagen/texto
class ListaOfertasViaje(BaseModel):
    ofertas: List[OfertaViaje] = Field(description="Lista de todas las ofertas de viaje o paquetes encontrados")

class FlyerPayload(BaseModel):
    text_content: Optional[str] = ""
    image_base64: Optional[str] = None
    mime_type: Optional[str] = "image/jpeg"

class ChatMessage(BaseModel):
    role: str  # "user" o "model"
    content: str

class ChatRequest(BaseModel):
    user_message: str
    travel_catalog: List[dict]  # Catálogo filtrado desde Node.js (solo activos y con cupo)
    history: Optional[List[ChatMessage]] = []

class ConfirmAction(str, Enum):
    APROBAR = "APROBAR"
    RECHAZAR = "RECHAZAR"
    DESCONOCIDO = "DESCONOCIDO"

class ConfirmResult(BaseModel):
    accion: ConfirmAction
    notas: Optional[str] = None

class ConfirmRequest(BaseModel):
    admin_message: str

# ------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------------
def optimizar_flyer_para_vision(imagen_bytes: bytes, max_size: int = 1024) -> Image.Image:
    img = Image.open(io.BytesIO(imagen_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

async def generate_with_fallback(contents, system_instruction, response_schema=None):
    last_error = None
    
    for model_name in MODELS_FALLBACK:
        for attempt in range(2):
            try:
                log.info("GEMINI_CALL", f"Invocando {model_name} (Intento {attempt + 1})...")
                start_time = time.time()
                
                config_args = {
                    "system_instruction": system_instruction,
                    "temperature": 0.2
                }
                if response_schema:
                    config_args["response_mime_type"] = "application/json"
                    config_args["response_schema"] = response_schema

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_args)
                )
                
                duration = round((time.time() - start_time) * 1000, 2)
                log.success("GEMINI_RES", f"Respuesta obtenida con {model_name} en {duration}ms")
                return response

            except Exception as e:
                last_error = e
                err_str = str(e)
                log.warn("GEMINI_RETRY", f"Falla con {model_name}: {err_str[:120]}... Reintentando.")
                
                if "503" in err_str or "429" in err_str:
                    await asyncio.sleep(1.5)
                else:
                    break
                    
    raise HTTPException(status_code=503, detail=f"Servicios de Google saturados. Último error: {str(last_error)}")

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------

@app.get("/health")
async def health_check():
    log.info("HEALTH", "Healthcheck invocado.")
    return {"status": "OK", "service": "Python AI Microservice (Gemini)"}

# 1. Extracción Multimodal desde Flyer
@app.post("/agent/extract-flyer", dependencies=[Depends(verify_internal_key)])
async def extract_flyer(payload: FlyerPayload):
    log.info("FLYER_REQ", "Payload recibido para extracción múltiple de flyer...")

    contents = []

    if payload.image_base64:
        try:
            image_bytes = base64.b64decode(payload.image_base64)
            img_pil = optimizar_flyer_para_vision(image_bytes)
            contents.append(img_pil)
        except Exception as e:
            log.error("IMAGE_PROC_ERR", f"Error procesando imagen Base64: {str(e)}")

    if payload.text_content:
        contents.append(f"Texto del mensaje/flyer:\n{payload.text_content}")

    if not contents:
        raise HTTPException(status_code=400, detail="No se proporcionó información para extraer.")

    try:
        log.info("GEMINI_CALL", "Invocando a Gemini 2.5 Flash (Structured Output Múltiple)...")

        system_instruction = (
            "Analiza minuciosamente la información proporcionada (texto e/o imagen). "
            "Es común que un afiche contenga MÚLTIPLES viajes, destinos o promociones distintas. "
            "Extrae TODAS y cada una de las ofertas de viaje individuales que identifiques."
        )

        response = await generate_with_fallback(
            contents=contents,
            system_instruction=system_instruction,
            response_schema=ListaOfertasViaje # <-- Usamos la lista como esquema
        )

        datos_dict = json.loads(response.text)
        return {"ok": True, "datos_extraidos": datos_dict}

    except Exception as e:
        log.error("GEMINI_ERR", f"Error procesando extracción con Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analizando con Gemini: {str(e)}")

# 2. Asistente Conversacional RAG
@app.post("/agent/chat", dependencies=[Depends(verify_internal_key)])
async def chat_agent(payload: ChatRequest):
    log.info("CHAT_REQ", f"Consulta de cliente recibida: '{payload.user_message}'", {
        "travel_catalog_count": len(payload.travel_catalog),
        "history_count": len(payload.history)
    })

    try:
        system_instruction = f"""
        Eres un asesor de viajes por WhatsApp atento y comercial.
        
        Tus reglas:
        1. Responde ÚNICAMENTE basándote en este catálogo de viajes disponibles que TIENEN STOCK:
        {payload.travel_catalog}
        
        2. Menciona claramente el destino, fecha, detalles y si quedan POCOS cupos (menciona los cupos si son < 3).
        3. Si el cliente pregunta por un destino o fecha que NO figura en este catálogo, infórmale amablemente que las plazas para ese viaje están agotadas o que no hay salidas disponibles por el momento, e insítalo a consultar por las otras opciones activas.
        4. Mantén respuestas concisas, amables y adecuadas para un chat de WhatsApp.
        """

        contents = []
        for msg in payload.history:
            role = "user" if msg.role.lower() in ["user", "human"] else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
        
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=payload.user_message)]))

        start_gemini = time.time()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )

        duration = round((time.time() - start_gemini) * 1000, 2)
        log.success("CHAT_RES", f"Respuesta de IA generada en {duration}ms:", response.text)

        return {"response": response.text}

    except Exception as e:
        log.error("CHAT_ERR", f"Error en Chat Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en Chat Gemini: {str(e)}")


# 3. Parser de Confirmación Humana
@app.post("/agent/confirm", response_model=ConfirmResult, dependencies=[Depends(verify_internal_key)])
async def parse_admin_confirmation(payload: ConfirmRequest):
    log.info("CONFIRM_REQ", f"Evaluando respuesta de admin: '{payload.admin_message}'")

    try:
        prompt = f"Determina si el siguiente mensaje del administrador autoriza la publicación ('APROBAR') o la rechaza ('RECHAZAR'):\n'{payload.admin_message}'"
        
        start_gemini = time.time()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ConfirmResult,
            )
        )

        duration = round((time.time() - start_gemini) * 1000, 2)
        log.success("CONFIRM_RES", f"Decisión interpretada en {duration}ms:", response.text)

        return ConfirmResult.model_validate_json(response.text)

    except Exception as e:
        log.error("CONFIRM_ERR", f"Error en Confirmación Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en Confirmación Gemini: {str(e)}")