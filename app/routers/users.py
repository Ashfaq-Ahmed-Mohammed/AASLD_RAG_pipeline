from app.database.database import SessionDep
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import select
from app.database.database import User
from app.utilities import get_password_hash
from app.models.models import UserRead, UserCreate

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", status_code=status.HTTP_201_CREATED, response_model=UserRead)
def create_user(user: UserCreate, session: SessionDep):
    try:
        hashed_password = get_password_hash(user.password)
        user_db = User(email=user.email, password=hashed_password)
        session.add(user_db)
        session.commit()
        session.refresh(user_db)
        return user_db
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User could not be created.")
    
@router.get("/{user_id}", response_model=UserRead)
def read_user(user_id: int, session: SessionDep):
    user = session.exec(select(User).where(User.id == user_id)).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return user
    


    



