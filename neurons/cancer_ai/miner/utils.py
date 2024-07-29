import uuid

def is_valid_uuid(uuid_to_test):
    if not uuid_to_test:
        return False
    try:
        uuid_obj = uuid.UUID(uuid_to_test)
        return str(uuid_obj) == uuid_to_test
    except ValueError:
        return False
