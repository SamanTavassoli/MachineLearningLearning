i = 0
def section(*args):
    global i
    title = ''
    i = i + 1
    if args:
        title = ' - ' + args[0]
    print('\n############## Section ' + str(i) + title + ' ##############\n')