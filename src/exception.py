import os
import sys
from src.logger import logging

def get_error_message(error_message,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    frame = exc_tb.tb_frame.f_code.co_filename
    message = "the error occured in file name [{0}] at line no [{1}] and the error message is [{2}]".format(frame,exc_tb.tb_lineno,str(error_message))
    return message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = get_error_message(error_message=error_message,error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
    
