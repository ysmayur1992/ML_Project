import os
import logging
from datetime import datetime

log_file = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),'Recent_Logs',log_file)

os.makedirs(log_path,exist_ok=True)

Log_File_Path = os.path.join(log_path,log_file)

logging.basicConfig(
    filename=Log_File_Path,
    format="[%(asctime)s] - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)