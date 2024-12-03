from fastapi import FastAPI
from fastapi.responses import Response
from pathlib import Path
from PIL import Image
import os
import zipfile
import io

image_dir = Path(__file__).parent
app = FastAPI()


def zipfiles(filenames):
    zip_filename = "archive.zip"

    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    for fpath in filenames:
        # Calculate path for file in zip
        fdir, fname = os.path.split(fpath)

        # Add file, at correct path
        zf.write(fpath, fname)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(
        s.getvalue(),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment;filename={zip_filename}"},
    )

    return resp


@app.get("/api/search")
async def root(query: str):
    filenames = [image_dir / f"test{i+1}.jpg" for i in range(6)]
    res = zipfiles(filenames)
    return res
