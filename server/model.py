from sqlalchemy import Column, Integer, String
from pydantic import BaseModel
# from db_config import Base
# from db_config import ENGINE


class UserTable(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(30), nullable=True)
    age = Column(Integer)

class CityTable(Base):
    __tablename__ = 'city'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(30), nullable=True)
    timezone = Column(String(30), nullable=True)


class User(BaseModel):
    id   : int
    name : str
    age  : int

class City(BaseModel):
    id      : int
    name    : str
    timezone: str


# def main():
#     # Table 없으면 생성
#     Base.metadata.create_all(bind=ENGINE)

if __name__ == "__main__":
    main()