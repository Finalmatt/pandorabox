About resumeTextExtracter

This program first goal was to extract brut text from resumes but it can extract text from any pdf file.
It is coded in Python. 
First the program convert the pdf file into image file, then it use the tesseract library to recognize text from it. At last it extract the text to a basic text file.
You have to create a folder(or use CV2) with all your pdf that you want to mine, the text files will be extracted in this folder.
The tesseract library is needed to execute the code. You have also to install the additional library which are 'imported' in the python code.
