import os
import io
from enum import Enum
from typing import Optional, List
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException, Security, Depends, status
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
    image_url: Optional[str] = None
    zernio_api_key: Optional[str] = None  # Por si se envía desde Node.js

@app.post("/agent/extract-flyer")
async def extract_flyer(payload: FlyerPayload, x_api_key: str = Header(None)):
    # 1. Validar API Key interna entre Node y Python
    if x_api_key != os.getenv("INTERNAL_API_KEY", "mi_clave_super_secreta_node_python_2026"):
        raise HTTPException(status_code=401, detail="X-API-Key interna inválida")

    api_key_zernio = payload.zernio_api_key or ZERNIO_API_KEY

    contents = []
    
    # 2. Descargar la imagen de Zernio usando la cabecera Bearer
    if payload.image_url:
        try:
            headers = {}
            # Si la URL viene de zernio.com, adjuntamos la API Key de Zernio
            if "zernio.com" in payload.image_url and api_key_zernio:
                headers["Authorization"] = f"Bearer {api_key_zernio}"

            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(payload.image_url, headers=headers, timeout=15.0)
                
                if resp.status_code == 200:
                    img = Image.open(io.BytesIO(resp.content))
                    contents.append(img)
                else:
                    print(f"⚠️ No se pudo descargar la imagen (Status {resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"⚠️ Error intentando descargar la imagen: {str(e)}")
            # Si hay texto disponible, no frenamos el proceso; continuamos con el texto

    # Si no hay imagen cargada ni texto, lanzamos error limpio
    if not contents and not payload.text_content:
        raise HTTPException(status_code=400, detail="No se proporcionó imagen válida ni texto para analizar.")

    # 3. Prompt para Gemini 2.5 Flash
    prompt = """
    Analiza la información proporcionada (texto y/o imagen de afiche/flyer) y extrae los datos de los viajes.
    Devuelve ÚNICAMENTE un objeto JSON válido con la siguiente estructura:
    {
      "destino": "Nombre del destino o ciudad",
      "fecha_salida": "Fecha o mes de salida",
      "precio": "Precio con moneda",
      "cupos": 10,
      "descripcion": "Resumen breve de lo que incluye",
      "contacto": "Teléfono o email de contacto"
    }
    """

    texto_final = f"{prompt}\n\nTexto recibido del mensaje:\n{payload.text_content or 'Sin texto acompañante'}"
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