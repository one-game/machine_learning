#! coding=utf-8
##################################
#
# 2018年9月1日
# one_game
# @todo: matrix calc
##################################
import numpy as np

state_map = {0: "健康", 1:"发烧"}
observation_map = {0: "正常", 1: "冷" , 2:"头晕"}

class Viterbi:
    """
     description: 
     there are m hidden state and n observations
     the priory dist. of hidden state is prio_probs with shape (1*m)
     the transfer probas. of hidden state is trans_probs with shape (m*m) 
        each row indicates p(col_index|row_index)
     the probas of p(observation|hidden state) is emit_probs with shape (m * n)
        each row indicates the probas of observations happen when a certain hidden state
    """

    def __init__(self, prio_probs, trans_probs, emit_probs):
        self.__prio_probs = prio_probs
        self.__trans_probs = trans_probs
        self.__emit_probs = emit_probs
    
    def calc_best(self, observations):
        num_step = len(observations)
        num_state = len(self.__prio_probs)
        #print num_step
        best_pathes = [ [-1] * num_state ] * num_step
        best_probs = np.ndarray((num_step,num_state))
        best_probs[0] = self.__prio_probs * self.__emit_probs.T[observations[0]]
        #print "0", best_probs[0]
        for step in range(1,num_step):
            observation = observations[step]
            #print observation_map[observation],"=========="
            last_state_probs = best_probs[step - 1] #copy.deepcopy(best_probs[step - 1])
            for state_i in range(num_state):
                #print state_map[state_i]
                #print "last prob :", best_probs[step - 1]
                probs = ((last_state_probs * self.__trans_probs.T[state_i]) * self.__emit_probs.T[observation][state_i])
                #print last_state_probs, self.__trans_probs.T[state_i] , self.__emit_probs.T[observation][state_i]
                #print "cur prob :", probs
                best_pathes[step][state_i] = np.argmax(probs)
                best_probs[step][state_i] = probs.max()
                #print last_state_probs
            # print "most prob stat is ", state_map[np.argmax(best_probs[step])]
            # print step, best_probs[step]
        best_last_state = np.argmax(best_probs[-1])
        best_path = [best_last_state]
        for path_ind in range(num_step - 1, 0, -1):
            last_best_state = best_pathes[path_ind][best_last_state]
            best_path.append(last_best_state)
        best_path.reverse()
        best_path = list(map(state_map.get, best_path))
        return best_path, best_probs[num_step - 1]



if __name__ == "__main__":

    prio_probs = np.array([0.6, 0.4])
    trans_probs = np.array([[0.7, 0.3], \
                            [0.4, 0.6]])
    emit_probs = np.array([ 
        [0.5,0.4,0.1],
        [0.1,0.3,0.6]
    ])

    viterbi = Viterbi(prio_probs, trans_probs, emit_probs)
    best_path, _ = viterbi.calc_best([0,1,2,2,2,0])
    print "=>".join(best_path)
