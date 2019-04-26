def split_show_views(line):
    show,views = line.split(',')
    return(show,views)
#print(split_show_views("hourly,21"))

def split_show_channel(line):
    show,channel = line.split(",")
    return (show, channel)
    
    
def extract_channel_views(show_views_channel):
    show = show_views_channel[0] 
    views,channel = show_views_channel[1][0],show_views_channel[1][1]
    return (channel, views)
print(extract_channel_views(('a',('b','c'))))  


def sum_counts(a, b):
    return a+b