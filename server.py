from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import uuid
from datetime import datetime, timedelta
import jwt
import asyncio
import re

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

SECRET_KEY = "refrigeracao-app-secret-key-2025"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
WARRANTY_DAYS = 90

security = HTTPBearer()
app = FastAPI(title="Sigma Refrigeração API")
api_router = APIRouter(prefix="/api")

USERS = {
    "Sigma": {"password": "Manin659*", "role": "ADM", "name": "Administrador"},
    "vitor": {"password": "Sigma", "role": "ADM2", "name": "Vitor"},
    "técnico": {"password": "Sigma", "role": "TECNICO", "name": "Técnico"},
    "tecnico": {"password": "Sigma", "role": "TECNICO", "name": "Técnico"},
}

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    username: str
    role: str
    name: str

# Padrão de TAG: XXX-SIG-0000 (3 letras, SIG fixo, 4 números)
TAG_PATTERN = re.compile(r'^[A-Z]{3}-SIG-\d{4}$')

class EquipmentCreate(BaseModel):
    tag: str
    local: str
    endereco: str
    nome_cliente: str
    telefone_cliente: str  # WhatsApp do cliente - OBRIGATÓRIO
    modelo: str
    marca: str
    numero_serie: str
    tipo_servico: str
    lembrete_preventiva: Optional[int] = None  # 3 or 6 months

    @field_validator('tag')
    @classmethod
    def validate_tag(cls, v):
        tag = v.upper()
        if not TAG_PATTERN.match(tag):
            raise ValueError('TAG deve seguir o padrão XXX-SIG-0000 (Ex: HOT-SIG-0001)')
        return tag
    
    @field_validator('telefone_cliente')
    @classmethod
    def validate_telefone(cls, v):
        # Remove caracteres não numéricos
        telefone = re.sub(r'\D', '', v)
        if len(telefone) < 10 or len(telefone) > 13:
            raise ValueError('Telefone deve ter entre 10 e 13 dígitos')
        return telefone

class ServiceRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tipo: str
    tecnico_nome: str
    descricao: str = ""
    pecas_trocadas: str = ""
    tipo_preventiva: str = ""
    data: datetime = Field(default_factory=datetime.utcnow)
    garantia_ate: Optional[datetime] = None

class ServiceRecordCreate(BaseModel):
    tipo: str
    tecnico_nome: str
    descricao: str = ""
    pecas_trocadas: str = ""
    tipo_preventiva: str = ""

class Equipment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tag: str = ""  # Campo opcional para compatibilidade com dados antigos
    local: str
    endereco: str
    nome_cliente: str
    telefone_cliente: str = ""  # WhatsApp do cliente
    modelo: str
    marca: str
    numero_serie: str
    tipo_servico: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = ""
    status: str = "ativo"
    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None
    historico_servicos: List[ServiceRecord] = []
    garantia_ate: Optional[datetime] = None
    lembrete_preventiva: Optional[int] = None
    proxima_preventiva: Optional[datetime] = None

class ClientGroup(BaseModel):
    nome_cliente: str
    equipments: List[Equipment]
    total_equipments: int

class Notification(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    equipment_id: str
    equipment_tag: str
    nome_cliente: str = ""
    telefone_cliente: str = ""
    tipo: str  # 'garantia_vencida' ou 'preventiva_vencida'
    mensagem: str
    data: datetime = Field(default_factory=datetime.utcnow)
    lida: bool = False

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return verify_token(credentials.credentials)

def is_admin(user: dict) -> bool:
    return user.get("role") in ["ADM", "ADM2"]

def is_main_admin(user: dict) -> bool:
    return user.get("role") == "ADM"

def calculate_warranty_date():
    return datetime.utcnow() + timedelta(days=WARRANTY_DAYS)

def calculate_next_preventive(months: int):
    return datetime.utcnow() + timedelta(days=months * 30)

async def check_preventive_reminders():
    while True:
        try:
            now = datetime.utcnow()
            
            # Verificar preventivas vencidas
            equipments_preventiva = await db.equipments.find({
                "status": "ativo",
                "proxima_preventiva": {"$lte": now}
            }).to_list(1000)
            
            for eq in equipments_preventiva:
                existing = await db.notifications.find_one({
                    "equipment_id": eq["id"],
                    "tipo": "preventiva_vencida",
                    "lida": False
                })
                if not existing:
                    notification = Notification(
                        equipment_id=eq["id"],
                        equipment_tag=eq.get("tag", ""),
                        nome_cliente=eq.get("nome_cliente", ""),
                        telefone_cliente=eq.get("telefone_cliente", ""),
                        tipo="preventiva_vencida",
                        mensagem=f"Manutenção preventiva vencida: {eq.get('tag', '')} - {eq.get('marca', '')} {eq.get('modelo', '')}"
                    )
                    await db.notifications.insert_one(notification.dict())
            
            # Verificar garantias vencidas
            equipments_garantia = await db.equipments.find({
                "status": "ativo",
                "garantia_ate": {"$lte": now}
            }).to_list(1000)
            
            for eq in equipments_garantia:
                existing = await db.notifications.find_one({
                    "equipment_id": eq["id"],
                    "tipo": "garantia_vencida",
                    "lida": False
                })
                if not existing:
                    notification = Notification(
                        equipment_id=eq["id"],
                        equipment_tag=eq.get("tag", ""),
                        nome_cliente=eq.get("nome_cliente", ""),
                        telefone_cliente=eq.get("telefone_cliente", ""),
                        tipo="garantia_vencida",
                        mensagem=f"Garantia vencida: {eq.get('tag', '')} - {eq.get('marca', '')} {eq.get('modelo', '')}"
                    )
                    await db.notifications.insert_one(notification.dict())
                    
        except Exception as e:
            logging.error(f"Error checking preventive reminders: {e}")
        await asyncio.sleep(3600)

async def cleanup_old_deleted_items():
    while True:
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=1)
            await db.equipments.delete_many({"status": "deleted", "deleted_at": {"$lt": cutoff_time}})
        except Exception as e:
            logging.error(f"Error in cleanup: {e}")
        await asyncio.sleep(3600)

@api_router.get("/")
async def root():
    return {"message": "Sigma Refrigeração API"}

@api_router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    user = USERS.get(request.username)
    if not user:
        raise HTTPException(status_code=401, detail="Usuário inválido")
    if user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Senha inválida")
    
    token = create_access_token({"username": request.username, "role": user["role"], "name": user["name"]})
    return LoginResponse(token=token, username=request.username, role=user["role"], name=user["name"])

@api_router.post("/equipments", response_model=Equipment)
async def create_equipment(equipment: EquipmentCreate, user=Depends(get_current_user)):
    # Check if tag already exists
    existing = await db.equipments.find_one({"tag": equipment.tag.upper(), "status": "ativo"})
    if existing:
        raise HTTPException(status_code=400, detail="Tag já existe")
    
    equipment_dict = equipment.dict()
    equipment_dict["tag"] = equipment_dict["tag"].upper()
    
    warranty_date = calculate_warranty_date()
    proxima_preventiva = None
    if equipment.lembrete_preventiva:
        proxima_preventiva = calculate_next_preventive(equipment.lembrete_preventiva)
    
    equipment_obj = Equipment(
        **equipment_dict,
        created_by=user["username"],
        garantia_ate=warranty_date,
        proxima_preventiva=proxima_preventiva
    )
    await db.equipments.insert_one(equipment_obj.dict())
    return equipment_obj

@api_router.post("/equipments/sync")
async def sync_equipments(equipments: List[EquipmentCreate], user=Depends(get_current_user)):
    """Sync offline equipments"""
    created = []
    for equipment in equipments:
        existing = await db.equipments.find_one({"tag": equipment.tag.upper(), "status": "ativo"})
        if not existing:
            equipment_dict = equipment.dict()
            equipment_dict["tag"] = equipment_dict["tag"].upper()
            warranty_date = calculate_warranty_date()
            proxima_preventiva = None
            if equipment.lembrete_preventiva:
                proxima_preventiva = calculate_next_preventive(equipment.lembrete_preventiva)
            
            equipment_obj = Equipment(
                **equipment_dict,
                created_by=user["username"],
                garantia_ate=warranty_date,
                proxima_preventiva=proxima_preventiva
            )
            await db.equipments.insert_one(equipment_obj.dict())
            created.append(equipment_obj)
    return {"synced": len(created), "equipments": created}

@api_router.get("/equipments", response_model=List[Equipment])
async def get_equipments(user=Depends(get_current_user)):
    equipments = await db.equipments.find({"status": "ativo"}).sort("created_at", -1).to_list(1000)
    return [Equipment(**eq) for eq in equipments]

@api_router.get("/equipments/grouped", response_model=List[ClientGroup])
async def get_equipments_grouped(user=Depends(get_current_user)):
    pipeline = [
        {"$match": {"status": "ativo"}},
        {"$sort": {"created_at": -1}},
        {"$group": {"_id": "$nome_cliente", "equipments": {"$push": "$$ROOT"}, "total_equipments": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    groups = await db.equipments.aggregate(pipeline).to_list(1000)
    return [ClientGroup(nome_cliente=g["_id"], equipments=[Equipment(**eq) for eq in g["equipments"]], total_equipments=g["total_equipments"]) for g in groups]

@api_router.get("/equipments/search")
async def search_equipments(q: str, user=Depends(get_current_user)):
    """Pesquisar equipamentos por TAG ou nome do cliente"""
    if not q or len(q) < 2:
        return []
    
    # Buscar por TAG ou nome do cliente (case insensitive)
    query = {
        "status": "ativo",
        "$or": [
            {"tag": {"$regex": q, "$options": "i"}},
            {"nome_cliente": {"$regex": q, "$options": "i"}}
        ]
    }
    equipments = await db.equipments.find(query).sort("created_at", -1).to_list(100)
    return [Equipment(**eq) for eq in equipments]

@api_router.get("/equipments/trash", response_model=List[Equipment])
async def get_trash(user=Depends(get_current_user)):
    if not is_main_admin(user):
        raise HTTPException(status_code=403, detail="Acesso negado")
    equipments = await db.equipments.find({"status": "deleted"}).sort("deleted_at", -1).to_list(1000)
    return [Equipment(**eq) for eq in equipments]

@api_router.get("/notifications", response_model=List[Notification])
async def get_notifications(user=Depends(get_current_user)):
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="Acesso negado")
    notifications = await db.notifications.find({"lida": False}).sort("data", -1).to_list(100)
    return [Notification(**n) for n in notifications]

@api_router.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str, user=Depends(get_current_user)):
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="Acesso negado")
    await db.notifications.update_one({"id": notification_id}, {"$set": {"lida": True}})
    return {"message": "Notificação marcada como lida"}

@api_router.get("/equipments/{equipment_id}", response_model=Equipment)
async def get_equipment(equipment_id: str, user=Depends(get_current_user)):
    equipment = await db.equipments.find_one({"id": equipment_id})
    if not equipment:
        raise HTTPException(status_code=404, detail="Equipamento não encontrado")
    return Equipment(**equipment)

@api_router.get("/equipments/tag/{tag}")
async def get_equipment_by_tag(tag: str):
    """Public endpoint for QR code scanning - read only"""
    equipment = await db.equipments.find_one({"tag": tag.upper(), "status": "ativo"})
    if not equipment:
        raise HTTPException(status_code=404, detail="Equipamento não encontrado")
    eq = Equipment(**equipment)
    return {
        "tag": eq.tag,
        "cliente": eq.nome_cliente,
        "local": eq.local,
        "endereco": eq.endereco,
        "marca": eq.marca,
        "modelo": eq.modelo,
        "numero_serie": eq.numero_serie,
        "tipo_servico": eq.tipo_servico,
        "garantia_ate": eq.garantia_ate,
        "historico_servicos": [{
            "tipo": s.tipo,
            "tecnico": s.tecnico_nome,
            "data": s.data,
            "descricao": s.descricao if s.tipo == "manutencao" else s.tipo_preventiva,
            "pecas": s.pecas_trocadas
        } for s in eq.historico_servicos]
    }

@api_router.get("/equipments/by-client/{client_name}")
async def get_equipments_by_client(client_name: str):
    """Public endpoint for group QR code - returns all equipments for a client"""
    from urllib.parse import unquote
    decoded_name = unquote(client_name)
    equipments = await db.equipments.find({
        "nome_cliente": {"$regex": f"^{decoded_name}$", "$options": "i"},
        "status": "ativo"
    }).to_list(1000)
    
    if not equipments:
        raise HTTPException(status_code=404, detail="Nenhum equipamento encontrado para este cliente")
    
    return {
        "cliente": decoded_name,
        "total": len(equipments),
        "equipamentos": [{
            "tag": eq.get("tag", ""),
            "marca": eq.get("marca", ""),
            "modelo": eq.get("modelo", ""),
            "local": eq.get("local", ""),
            "numero_serie": eq.get("numero_serie", ""),
            "garantia_ate": eq.get("garantia_ate"),
        } for eq in equipments]
    }

@api_router.put("/equipments/{equipment_id}", response_model=Equipment)
async def update_equipment(equipment_id: str, equipment: EquipmentCreate, user=Depends(get_current_user)):
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="Acesso negado")
    
    existing = await db.equipments.find_one({"id": equipment_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Equipamento não encontrado")
    
    # Check if new tag conflicts with another equipment
    if equipment.tag.upper() != existing.get("tag", "").upper():
        tag_exists = await db.equipments.find_one({"tag": equipment.tag.upper(), "status": "ativo", "id": {"$ne": equipment_id}})
        if tag_exists:
            raise HTTPException(status_code=400, detail="Tag já existe")
    
    update_data = equipment.dict()
    update_data["tag"] = update_data["tag"].upper()
    
    if equipment.lembrete_preventiva:
        update_data["proxima_preventiva"] = calculate_next_preventive(equipment.lembrete_preventiva)
    
    await db.equipments.update_one({"id": equipment_id}, {"$set": update_data})
    updated = await db.equipments.find_one({"id": equipment_id})
    return Equipment(**updated)

@api_router.delete("/equipments/{equipment_id}")
async def delete_equipment(equipment_id: str, user=Depends(get_current_user)):
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="Acesso negado")
    
    existing = await db.equipments.find_one({"id": equipment_id, "status": "ativo"})
    if not existing:
        raise HTTPException(status_code=404, detail="Equipamento não encontrado")
    
    await db.equipments.update_one({"id": equipment_id}, {"$set": {"status": "deleted", "deleted_at": datetime.utcnow(), "deleted_by": user["username"]}})
    return {"message": "Movido para lixeira"}

@api_router.post("/equipments/{equipment_id}/restore")
async def restore_equipment(equipment_id: str, user=Depends(get_current_user)):
    if not is_main_admin(user):
        raise HTTPException(status_code=403, detail="Acesso negado")
    
    existing = await db.equipments.find_one({"id": equipment_id, "status": "deleted"})
    if not existing:
        raise HTTPException(status_code=404, detail="Não encontrado")
    
    await db.equipments.update_one({"id": equipment_id}, {"$set": {"status": "ativo", "deleted_at": None, "deleted_by": None}})
    return {"message": "Restaurado"}

@api_router.delete("/equipments/{equipment_id}/permanent")
async def permanent_delete(equipment_id: str, user=Depends(get_current_user)):
    if not is_main_admin(user):
        raise HTTPException(status_code=403, detail="Acesso negado")
    
    result = await db.equipments.delete_one({"id": equipment_id, "status": "deleted"})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Não encontrado")
    return {"message": "Excluído permanentemente"}

@api_router.post("/equipments/{equipment_id}/services", response_model=ServiceRecord)
async def add_service(equipment_id: str, service: ServiceRecordCreate, user=Depends(get_current_user)):
    existing = await db.equipments.find_one({"id": equipment_id, "status": "ativo"})
    if not existing:
        raise HTTPException(status_code=404, detail="Equipamento não encontrado")
    
    warranty_date = calculate_warranty_date()
    service_record = ServiceRecord(
        tipo=service.tipo,
        tecnico_nome=service.tecnico_nome,
        descricao=service.descricao,
        pecas_trocadas=service.pecas_trocadas,
        tipo_preventiva=service.tipo_preventiva,
        garantia_ate=warranty_date
    )
    
    update_data = {
        "$push": {"historico_servicos": service_record.dict()},
        "$set": {"garantia_ate": warranty_date}
    }
    
    # Reset preventive reminder if configured
    if existing.get("lembrete_preventiva"):
        update_data["$set"]["proxima_preventiva"] = calculate_next_preventive(existing["lembrete_preventiva"])
    
    await db.equipments.update_one({"id": equipment_id}, update_data)
    
    # Clear any pending preventive notification
    await db.notifications.delete_many({"equipment_id": equipment_id, "tipo": "preventiva_vencida", "lida": False})
    
    return service_record

app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@app.on_event("startup")
async def startup():
    asyncio.create_task(cleanup_old_deleted_items())
    asyncio.create_task(check_preventive_reminders())

@app.on_event("shutdown")
async def shutdown():
    client.close()
