#Download deepforest before tests start
from deepforest import main

def pytest_sessionstart():
    # prepare something ahead of all tests
    m = main.deepforest()
    m.use_release()    
