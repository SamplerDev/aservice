import os
import io
from enum import Enum
from typing import Optional, List
import requests
import base64
from PIL import Image
from fastapi import FastAPI, HTTPException, Security, Depends, status, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv





load_dotenv()

app = FastAPI(title="AI Agentic Microservice (Gemini)")

# Inicializar cliente oficial de Google GenAI
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
print(INTERNAL_API_KEY)
# ------------------------------------------------------------------
# ESQUEMAS PYDANTIC (Structured Outputs)
# ------------------------------------------------------------------

class OfertaViaje(BaseModel):
    destino: str = Field(description="Ciudad, región o país principal del viaje")
    fecha_salida: str = Field(description="Fecha en formato YYYY-MM-DD o aproximada")
    descripcion: str = Field(description="Resumen de inclusiones, estadía, transporte y precio")
    contacto: Optional[str] = Field(None, description="Teléfono o medio de contacto extraído")
    cupos: Optional[int] = Field(1, description="Cantidad estimada de cupos o lugares disponibles mencionados")

class ExtractRequest(BaseModel):
    image_url: str

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

async def verify_internal_key(api_key: str = Security(api_key_header)):
    if not INTERNAL_API_KEY or api_key != INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Acceso no autorizado: X-API-Key inválida o ausente"
        )


def optimizar_flyer_para_vision(imagen_bytes: bytes, max_size: int = 1024) -> bytes:
    img = Image.open(io.BytesIO(imagen_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "OK", "service": "Python AI Microservice (Gemini)"}

# 1. Extracción Multimodal desde Flyer
class FlyerPayload(BaseModel):
    text_content: Optional[str] = ""
    image_base64: Optional[str] = None
    mime_type: Optional[str] = "image/jpeg"

@app.post("/agent/extract-flyer")
async def extract_flyer(payload: FlyerPayload, x_api_key: str = Header(None)):
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key interna inválida")

    contents = []

    # 2. Convertir el Base64 directamente a un objeto PIL Image para Gemini
    if payload.image_base64:
        try:
            image_bytes = base64.b64decode(payload.image_base64)
            img = Image.open(io.BytesIO(image_bytes))
            contents.append(img)
            print("✅ Imagen Base64 cargada correctamente en Python.")
        except Exception as e:
            print(f"⚠️ Error al decodificar la imagen Base64: {e}")

    # 3. Prompt para Gemini 2.5 Flash
    prompt = """
    Analiza la información proporcionada (texto e/o imagen de afiche) y extrae los datos del viaje.
    Devuelve ÚNICAMENTE un objeto JSON con la siguiente estructura:
    {
      "destino": "string o null",
      "fecha_salida": "string o null",
      "precio": "string o null",
      "cupos": integer_o_null,
      "descripcion": "resumen breve de lo que incluye",
      "contacto": "string o null"
    }
    """

    texto_final = f"{prompt}\n\nTexto recibido:\n{payload.text_content or 'Sin texto acompañante'}"
    contents.append(texto_final)

    # 4. Invocación a Gemini 2.5 Flash
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(contents)
        return {"ok": True, "datos_extraidos": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando con Gemini: {str(e)}")


# 2. Asistente Conversacional RAG
@app.post("/agent/chat", dependencies=[Depends(verify_internal_key)])
async def chat_agent(payload: ChatRequest):
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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Chat Gemini: {str(e)}")

# 3. Parser de Confirmación Humana
@app.post("/agent/confirm", response_model=ConfirmResult, dependencies=[Depends(verify_internal_key)])
async def parse_admin_confirmation(payload: ConfirmRequest):
    try:
        prompt = f"Determina si el siguiente mensaje del administrador autoriza la publicación ('APROBAR') o la rechaza ('RECHAZAR'):\n'{payload.admin_message}'"
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ConfirmResult,
            )
        )
        return ConfirmResult.model_validate_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Confirmación Gemini: {str(e)}")