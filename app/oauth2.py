import jwt
from jwt.exceptions import InvalidTokenError
from app.config import settings
from app.models.models import TokenData, Token
from datetime import datetime, timedelta, timezone
from app.database.database import SessionDep
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import select
from app.database.database import User
from fastapi import Depends, HTTPException, status

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

secret_key = settings.secret_key
algorithm = settings.algorithm
expiry = settings.access_token_expire_minutes

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expiry)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt

def verify_access_token(token: str, credentials_exception) -> TokenData:
    try:
        paylaod = jwt.decode(token, secret_key, algorithms=[algorithm])
        id: str = paylaod.get("sub")

        if id is None:
            raise credentials_exception
        token_data = TokenData(id=str(id))
    except jwt.PyJWTError:
        raise credentials_exception
    
    return token_data

def get_current_user(token: str = oauth2_scheme, session: SessionDep = None):
    credentials_exception = HTTPException(status = status.HTTP_401_UNAUTHORIZED,
                                          detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    token_data = verify_access_token(token, credentials_exception)
    user = session.exec(select(User).where(User.id == token_data.id)).first()
    if user is None:
        raise credentials_exception
    return user
   

