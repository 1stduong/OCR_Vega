import numpy as np
import cv2
import os
from natsort import natsorted
import easyocr
from jiwer import wer, cer
import time
import psutil
import pandas as pd
from math import sqrt
import pytesseract
from pytesseract import Output
from difflib import SequenceMatcher