import io
import base64
import tempfile

def string_to_file(string):
    file_like_obj = tempfile.NamedTemporaryFile()
    file_like_obj.write(string)
    file_like_obj.flush()
    file_like_obj.seek(0)
    return file_like_obj

def file2b64(file_like):
    with open(file_like, 'rb') as f:
        s = f.read()
        b64 = base64.b64encode(s)
        s64 = str(b64, encoding="utf-8")
    return s64

