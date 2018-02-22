import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is a size N array of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    list1=[]
    dp_table = [[float('-inf')for y in range(N)] for x in range(L)]
    bp_table = [[float('-inf')for y in range(N)] for x in range(L)]
    #base case for the recursion
    for i in range(0,L):
        list1.append(emission_scores[0][i] + start_scores[i])

    for cur in range(0,L):
        dp_table[cur][0] = list1[cur]

    #Now for the recursive step, that maximise over incoming transitions reusing the best incoming score.
    for t in range(1, N):
        for tag2 in range(L):
            dp_table[tag2][t] = float('-inf')
            for tag1 in range(L):
                score = dp_table[tag1][t-1] + trans_scores[tag1][tag2]
                if score > dp_table[tag2][t]:
                    dp_table[tag2][t] = score
                    bp_table[tag2][t] = tag1
            dp_table[tag2][t] += emission_scores[t][tag2]

    flag=0
    b=0
    maximum=float('-inf')
    for p in range(L):
        temp = dp_table[p][N-1] + end_scores[p]
        flag = 1 if temp > maximum else 0
        if flag==1:
            b = p
            maximum = temp
    #print "Maximum Answer is: ", maximum, "at: ", b
    final = b
    y = [final]
    #print y
    for i in range(N-1,0,-1):
        temp2=bp_table[final][i]
        y.append(temp2)
        final=temp2
    y.reverse()
    return (maximum, y)

