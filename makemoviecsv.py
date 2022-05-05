import re
from functools import reduce
import csv
import tqdm  
import random

'''
    1. Read from full_qa_dev.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''
'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''
def gather_dataset():
    questions = []; answers = []

# enter full path of the file with training data
    lines = open('transformerdemo/full_qa_dev.txt').read()


    for line in lines.splitlines():
        c =line.split('?')
        tups = tuple(c)
        if len(tups)==2:
            questions.append(tups[0])
            answers.append(tups[1])
            
           


    return questions,answers

textandanswers=[]
def saveMessages(questions,answers):
    with open('conversations.csv', 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerow([ 'Input' ,'Target'])
    
            for  index, question  in enumerate(questions) :
                if  (index < 16000):
                    writer.writerow([questions[index], answers[index]])
                else:
                    break
               
			
                                      
  



questions, answers = gather_dataset()
print (questions[-1])
print( answers [-1])

print ('>> gathered questions and answers.\n')
saveMessages(questions,answers)

