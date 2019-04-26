def split_fileA(line):
    # split the input line in word and count on the comma
    word, count = line.split(',',1)
    # turn the count to an integer  
    count = int(count)
    return (word, count)
    
def split_fileB(line):
    # split the input line into word, date and count_string
    date_word,count_string = line.split(',')
    date,word = date_word.split(' ')
    return (word, date + " " + count_string) 
    
print(split_fileB('Jan-01 able,5'))

print('Jan-01   able'.split(' ',1))
#print('Jan-01 able,5'.split(',',1))
    