"""
Automatic encryption/decryption for Bittensor hotkeys using unencrypted hotkey files.
"""

import os
import json
import hashlib
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.backends import default_backend
import bittensor as bt
CHUNK_SIZE = 64 * 1024 * 1024  # 64MB chunks for file I/O

def load_hotkey_from_file(wallet_name: str, hotkey_name: str = "default") -> bt.Keypair:
    """
    Load hotkey directly from unencrypted JSON file without password.
    
    Args:
        wallet_name: Name of the wallet
        hotkey_name: Name of the hotkey (default: "default")
    
    Returns:
        Bittensor Keypair object
    """

    hotkey_path = Path.home() / ".bittensor" / "wallets" / wallet_name / "hotkeys" / hotkey_name
    
    if not hotkey_path.exists():
        raise FileNotFoundError(f"Hotkey file not found: {hotkey_path}")
    
    with open(hotkey_path, 'r', encoding='utf-8') as f:
        hotkey_data = json.load(f)
    
    # Extract private key from JSON
    private_key_hex = hotkey_data["privateKey"].replace("0x", "")
    
    # Create Keypair from private key
    return bt.Keypair.create_from_private_key(private_key_hex)


def encrypt_file_for_hotkey(
    file_path: str,
    target_hotkey_ss58: str,
    output_path: str = None,
    chunk_size: int = CHUNK_SIZE
    )->str:

    """
    Encrypt a file for a specific validator hotkey.
    
    Args:
        file_path: Path to file to encrypt
        target_hotkey_ss58: SS58 address of validator who can decrypt
        output_path: Where to save encrypted file (optional)
    
    Returns:
        Path to encrypted file
    """

    try:
        bt.Keypair(ss58_address=target_hotkey_ss58)
    except Exception as e:
        raise ValueError(f"Invalid SS58 address: {target_hotkey_ss58}") from e
    
    encryption_key = os.urandom(32)
    iv = os.urandom(16)
    challenge = os.urandom(32)
    print("Generating encryption key, IV, and challenge...")
    
    print("Deriving key-wrapping key")
    key_wrap_key = hashlib.pbkdf2_hmac(
        'sha256',
        challenge,
        b'hotkey-symmetric-wrap',
        100000
    )
    print("Key derivation complete")
    
    key_cipher = Cipher(algorithms.AES(key_wrap_key), modes.CBC(iv), backend=default_backend())
    key_encryptor = key_cipher.encryptor()
    
    key_pad_length = 16 - (len(encryption_key) % 16)
    padded_key = encryption_key + bytes([key_pad_length] * key_pad_length)
    wrapped_key = key_encryptor.update(padded_key) + key_encryptor.finalize()
    
    hmac_calculator = hmac.HMAC(encryption_key, hashes.SHA256(), backend=default_backend())
    hmac_calculator.update(iv + wrapped_key)
    
    cipher = Cipher(algorithms.AES(encryption_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    print("Reading and encrypting file in chunks...")
    
    if output_path is None:
        output_path = f"{file_path}.encrypted"

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    BLOCK_SIZE = 64 * 1024
    temp_encrypted_file = output_path + '.tmp'
    
    with open(file_path, 'rb') as input_file, open(temp_encrypted_file, 'wb') as temp_file:
        buffer = b''
        
        while True:
            chunk = input_file.read(chunk_size)
            if not chunk:
                break

            buffer += chunk

            while len(buffer) >= BLOCK_SIZE:
                block = buffer[:BLOCK_SIZE]
                buffer = buffer[BLOCK_SIZE:]

                encrypted_block = encryptor.update(block)
                temp_file.write(encrypted_block)
                hmac_calculator.update(encrypted_block)
        
        while len(buffer) >= 16:
            block = buffer[:16]
            buffer = buffer[16:]

            encrypted_block = encryptor.update(block)
            temp_file.write(encrypted_block)
            hmac_calculator.update(encrypted_block)
        
        if buffer:
            pad_length = 16 - len(buffer)
            padded_buffer = buffer + bytes([pad_length] * pad_length)
            encrypted_block = encryptor.update(padded_buffer)
            temp_file.write(encrypted_block)
            hmac_calculator.update(encrypted_block)
        else:
            padding_block = bytes([16] * 16)
            encrypted_block = encryptor.update(padding_block)
            temp_file.write(encrypted_block)
            hmac_calculator.update(encrypted_block)
        
        final_block = encryptor.finalize()
        if final_block:
            temp_file.write(final_block)
            hmac_calculator.update(final_block)
    
    signature = hmac_calculator.finalize()
    
    print("Writing encrypted file...")
    
    with open(temp_encrypted_file, 'rb') as temp_file, open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write('{\n')
        output_file.write('  "version": 3,\n')
        output_file.write(f'  "recipient_ss58": "{target_hotkey_ss58}",\n')
        output_file.write(f'  "challenge": "{challenge.hex()}",\n')
        output_file.write(f'  "iv": "{iv.hex()}",\n')
        output_file.write(f'  "wrapped_key": "{wrapped_key.hex()}",\n')
        output_file.write('  "encrypted_data": "')
        
        hex_buffer_size = 64 * 1024
        while True:
            encrypted_chunk = temp_file.read(hex_buffer_size)
            if not encrypted_chunk:
                break
            hex_chunk = encrypted_chunk.hex()
            output_file.write(hex_chunk)
        
        output_file.write('",\n')
        output_file.write(f'  "hmac": "{signature.hex()}"\n')
        output_file.write('}\n')
    
    os.remove(temp_encrypted_file)
    print(f"Encryption complete! Saved to: {output_path}")
    
    return output_path


def decrypt_file_with_hotkey(
    encrypted_file_path: str, 
    wallet_name: str, 
    hotkey_name: str = "default"
    ) -> str:
    """
    Decrypt a file using validator's hotkey.
    
    Args:
        encrypted_file_path: Path to encrypted file
        wallet_name: Wallet name containing the hotkey
        hotkey_name: Hotkey name (default: "default")
    
    Returns:
        Path to decrypted file
    """
    hotkey = load_hotkey_from_file(wallet_name, hotkey_name)
    
    with open(encrypted_file_path, 'r', encoding='utf-8') as f:
        package = json.load(f)
    
    # Verify the file was encrypted for this validator
    if package['recipient_ss58'] != hotkey.ss58_address:
        raise ValueError(
            "Encrypted file is not intended for this hotkey. "
            "Only the intended recipient can decrypt this file."
        )

    # Extract components
    challenge = bytes.fromhex(package['challenge'])
    iv = bytes.fromhex(package['iv'])
    wrapped_key = bytes.fromhex(package['wrapped_key'])
    encrypted_data_hex = package['encrypted_data']
    expected_hmac = bytes.fromhex(package['hmac'])
    
    # Use private key to sign the challenge (proves ownership before decryption)
    challenge_signature = hotkey.sign(challenge)

    key_wrap_key = hashlib.pbkdf2_hmac(
        'sha256', 
        challenge, 
        b'hotkey-symmetric-wrap', 
        100000
    )


    if not hotkey.verify(challenge, challenge_signature):
        raise ValueError("Invalid signature - hotkey ownership verification failed")


    key_cipher = Cipher(algorithms.AES(key_wrap_key), modes.CBC(iv), backend=default_backend())
    key_decryptor = key_cipher.decryptor()
    decrypted_padded_key = key_decryptor.update(wrapped_key) + key_decryptor.finalize()

    pad_length = decrypted_padded_key[-1]
    encryption_key = decrypted_padded_key[:-pad_length]
    
    hmac_verifier = hmac.HMAC(encryption_key, hashes.SHA256(), backend=default_backend())
    hmac_verifier.update(iv + wrapped_key)
    
    cipher = Cipher(algorithms.AES(encryption_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    encrypted_path = Path(encrypted_file_path)
    decrypted_dir = encrypted_path.parent / "decrypted"
    decrypted_dir.mkdir(parents=True, exist_ok=True)
    
    if encrypted_path.name.endswith('.encrypted'):
        original_filename = encrypted_path.name[:-10]
    else:
        original_filename = encrypted_path.name.replace('.encrypted', '')
    
    output_path = str(decrypted_dir / original_filename)
    temp_decrypted_file = output_path + '.tmp'
    
    hex_chunk_size = 128 * 1024
    i = 0
    
    with open(temp_decrypted_file, 'wb') as temp_file:
        while i < len(encrypted_data_hex):
            hex_chunk = encrypted_data_hex[i:i+hex_chunk_size]
            
            if len(hex_chunk) % 2 != 0:
                hex_chunk = hex_chunk[:-1]
                i -= 1
            
            encrypted_bytes = bytes.fromhex(hex_chunk)
            hmac_verifier.update(encrypted_bytes)
            
            BLOCK_SIZE = 64 * 1024
            buffer = encrypted_bytes
            
            while len(buffer) >= BLOCK_SIZE:
                chunk = buffer[:BLOCK_SIZE]
                buffer = buffer[BLOCK_SIZE:]
                decrypted_chunk = decryptor.update(chunk)
                temp_file.write(decrypted_chunk)
            
            while len(buffer) >= 16:
                chunk = buffer[:16]
                buffer = buffer[16:]
                decrypted_chunk = decryptor.update(chunk)
                temp_file.write(decrypted_chunk)
            
            i += hex_chunk_size
        
        final_block = decryptor.finalize()
        if final_block:
            temp_file.write(final_block)
    
    try:
        hmac_verifier.verify(expected_hmac)
    except Exception as e:
        os.remove(temp_decrypted_file)
        raise ValueError("HMAC verification failed - file may be tampered with") from e
    
    with open(temp_decrypted_file, 'rb') as temp_file, open(output_path, 'wb') as output_file:
        temp_file.seek(0, 2)
        file_size = temp_file.tell()
        temp_file.seek(0)
        
        if file_size < 16:
            os.remove(temp_decrypted_file)
            raise ValueError("Decrypted file too small")
        
        temp_file.seek(-16, 2)
        last_block = temp_file.read(16)
        pad_length = last_block[-1]
        
        if pad_length > 16:
            os.remove(temp_decrypted_file)
            raise ValueError("Invalid padding length")
        
        temp_file.seek(0)
        bytes_to_write = file_size - pad_length
        
        remaining = bytes_to_write
        while remaining > 0:
            chunk_size = min(64 * 1024, remaining)
            chunk = temp_file.read(chunk_size)
            if not chunk:
                break
            output_file.write(chunk)
            remaining -= len(chunk)
    
    os.remove(temp_decrypted_file)
    print(f"Decryption complete! Saved to: {output_path}")
    
    return output_path





def list_available_hotkeys(wallet_name: str):
    """List all available hotkeys for a wallet."""
    hotkeys_dir = Path.home() / ".bittensor" / "wallets" / wallet_name / "hotkeys"
    
    if not hotkeys_dir.exists():
        print(f"Wallet '{wallet_name}' not found")
        return
    
    hotkeys = []
    for hotkey_file in hotkeys_dir.iterdir():
        if hotkey_file.is_file():
            try:
                with open(hotkey_file, 'r', encoding='utf-8') as f:
                    hotkey_data = json.load(f)
                hotkeys.append({
                    'name': hotkey_file.name,
                    'ss58': hotkey_data['ss58Address'],
                    'encrypted': 'encrypted' in hotkey_file.name.lower()
                })
            except (json.JSONDecodeError, KeyError, PermissionError):
                pass
    
    print(f"Available hotkeys in wallet '{wallet_name}':")