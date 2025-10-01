import keyring
import os

from enum import Enum

class TYPE_PASSWORD_SOURCE(Enum):
    KEYRING = "keyring"
    OS = "os"

def read_password_from_keychain(username, pw_key):
    return keyring.get_password(pw_key, username)

def read_password_from_os_environ(username, pw_key):
    return os.environ[f"{pw_key}_{username}"]

def read_password(username, pw_source, pw_key):
    assert pw_source in [source.value for source in TYPE_PASSWORD_SOURCE], f"Database password source {pw_source} is not supported yet."

    if pw_source.lower() == TYPE_PASSWORD_SOURCE.KEYRING:
        return read_password_from_keychain(username, pw_key)
    elif pw_source.lower() == TYPE_PASSWORD_SOURCE.OS:
        return read_password_from_os_environ(username, pw_key)
    else:
        raise NotImplementedError