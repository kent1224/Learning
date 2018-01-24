# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:17:52 2017

@author: 14224
"""

"""
Berkeley example
"""

class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        actions = []

        def max_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)

            v = max([(value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth),
                      action) for action in legal_moves], key=lambda x: x[0])
            actions.append(v[1])
            return v[0]

        def min_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            v = min([value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth)
                     for action in legal_moves])   #因為是對手，算出值就好，不用move
            return v

        def value(game_state, agent_num=0, depth=1):
            num_agents = game_state.getNumAgents()
            if agent_num >= num_agents:
                if game_state.isLose() or game_state.isWin():   # 輸 / 贏 / 沒有legal move了
                    return self.evaluationFunction(game_state)
                if self.depth > depth:
                    return value(game_state, 0, depth+1)
                else:                                           # 到達預設的深度了 
                    return self.evaluationFunction(game_state)

            if agent_num == 0:
                return max_value(game_state, agent_num, depth)
            else:
                return min_value(game_state, agent_num, depth)

        value(gameState)
        return actions[-1]


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        actions = []

        def max_value(game_state, agent_num, depth, alpha, beta):
            v = (-sys.maxsize, None)
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            for action in legal_moves:
                successor_state = game_state.generateSuccessor(agent_num, action)
                v = max(v, (value(successor_state, agent_num+1, depth, alpha, beta), action), key=lambda x: x[0])
                # 因為每個v[0]要檢測是否prune，所以max裡面沒有迴圈，也因為沒有迴圈，所以要跟原本的v比
                if v[0] > beta:          # 與udacity不一樣，先檢測是否prune
                    actions.append(v[1])
                    return v[0]
                alpha = max(alpha, v[0]) # 再更新alpha值
            actions.append(v[1])
            return v[0]

        def min_value(game_state, agent_num, depth, alpha, beta):
            v = sys.maxsize
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            for action in legal_moves:
                successor_state = game_state.generateSuccessor(agent_num, action)
                v = min(v, value(successor_state, agent_num+1, depth, alpha, beta))
                # 因為每個v[0]要檢測是否prune，所以max裡面沒有迴圈，也因為沒有迴圈，所以要跟原本的v比
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def value(game_state, agent_num=0, depth=1, alpha=-sys.maxsize, beta=sys.maxsize):
            num_agents = game_state.getNumAgents()
            if agent_num >= num_agents:
                if game_state.isLose() or game_state.isWin():
                    return self.evaluationFunction(game_state)
                if self.depth > depth:
                    return value(game_state, 0, depth+1, alpha, beta)
                else:
                    return self.evaluationFunction(game_state)

            if agent_num == 0:
                return max_value(game_state, agent_num, depth, alpha, beta)
            else:
                return min_value(game_state, agent_num, depth, alpha, beta)

        value(gameState)
        return actions[-1]

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        actions = []

        def max_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)

            v = max([(value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth),
                      action) for action in legal_moves], key=lambda x: x[0])
            actions.append(v[1])
            return v[0]

        def min_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            """ 
            The only place different from minimax.
            give probability to opponent's move.
            because to choose uniformly at random from their legal moves.
            """            
            v = sum([value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth)
                     for action in legal_moves])/len(legal_moves)
            return v

        def value(game_state, agent_num=0, depth=1):
            num_agents = game_state.getNumAgents()
            if agent_num >= num_agents:
                if game_state.isLose() or game_state.isWin():
                    return self.evaluationFunction(game_state)
                if self.depth > depth:
                    return value(game_state, 0, depth+1)
                else:
                    return self.evaluationFunction(game_state)

            if agent_num == 0:
                return max_value(game_state, agent_num, depth)
            else:
                return min_value(game_state, agent_num, depth)

        value(gameState)
        return actions[-1]


"""
Udacity example
"""

def minimax(self, game, depth, maximizing_player=True):  #預設第一個由我先開始
                                                         #depth當做counter來想
        # limit 1: time                                         
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()
        
        #limit 2 & 3: no legal move or run out of depth
        if not legal_moves or depth <= 0:
            return self.score(game, self), None

        best_move = None
        if maximizing_player:
            best_value = float("-inf")    # 先設定最大值是負無線大　
            for move in legal_moves:      # 進入迴圈，檢測這個node下面可能的值
                next_state = game.forecast_move(move)
                value, _ = self.minimax(next_state, depth - 1, False)
                if value > best_value:    #在迴圈中，找出可能的node中最大的值以及其move
                    best_value, best_move = value, move
        else:
            best_value = float("inf")     # 先設定最大值是無線大
            for move in legal_moves:      # 進入迴圈，檢測這個node下面可能的值
                next_state = game.forecast_move(move)
                value, _ = self.minimax(next_state, depth - 1, True)
                if value < best_value:    #在迴圈中，找出可能的node中最小的值以及其move
                    best_value, best_move = value, move

        return best_value, best_move

def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves or depth <= 0:
            return self.score(game, self), None

        best_move = None
        if maximizing_player:
            best_value = float("-inf")
            for move in legal_moves:
                next_state = game.forecast_move(move)
                value, _ = self.alphabeta(next_state, depth - 1, alpha, beta, False)
                alpha = max(alpha, value)    # 利用迴圈更新alpha值: max(既有的alpha, 另一個node的值)
                if value > best_value:       # 照常找出可能的node中最大的值以及其move
                    best_value, best_move = value, move

                if alpha >= beta:            # prune
                    break
        else:
            best_value = float("inf")
            for move in legal_moves:
                next_state = game.forecast_move(move)
                value, _ = self.alphabeta(next_state, depth - 1, alpha, beta, True)
                beta = min(beta, value)     # 利用迴圈更新beta值: min(既有的beta, 另一個node的值)
                if value < best_value:      # 照常找出可能的node中最小的值以及其move
                    best_value, best_move = value, move

                if alpha >= beta:           # prune
                    break
                
        return best_value, best_move
