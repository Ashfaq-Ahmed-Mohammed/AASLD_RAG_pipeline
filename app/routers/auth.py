import app.oauth2 as oauth2
import app.utilities as utils
from sqlmodel import select
from app.database.database import SessionDep, User
from app.models.models import UserLogin, Token
from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter(prefix="/login", tags=["Authentication"])

@router.post("/", response_model=Token)
def login(user_credentials: UserLogin, session: SessionDep = None):
    user_email = user_credentials.email
    user = select(User).where(User.email == user_email)
    user = session.exec(user).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Credentials test")

    if not utils.verify_password(user_credentials.password, user.password):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Credentials")
    
    access_token = oauth2.create_access_token(data={"sub": str(user.id)})

    return {"access_token": access_token, "token_type": "bearer"}
    


   

