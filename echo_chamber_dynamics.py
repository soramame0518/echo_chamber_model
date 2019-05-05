# coding: utf-8
# Echo Chamber Model
# echo_chamber_dynamics.py
# Last Update: 20190410
# by Kazutoshi Sasahara

import os
import numpy as np
import networkx as nx
import pandas as pd
from agent import Agent
from social_media import SocialMedia
import analysis


class EchoChamberDynamics(object):
    def __init__(self, num_agents, num_links, epsilon, sns_seed, l, data_dir):
        self.num_agents = num_agents
        self.l = l
        self.epsilon = epsilon
        self.set_agents(num_agents, epsilon)
        self.social_media = SocialMedia(num_agents, num_links, l, sns_seed)
        self.data_dir = data_dir
        self.opinion_data = []
        self.screen_diversity_data = []
        if not os.path.isdir(data_dir):
            os.makedirs(os.path.join(data_dir, 'data'))
            os.makedirs(os.path.join(data_dir, 'network_data'))

            
    def set_agents(self, num_agents, epsilon):
        screen_diversity = analysis.screen_diversity([], bins=10)
        self.agents = [Agent(i, epsilon, screen_diversity) for i in range(num_agents)]

        
    def total_discordant_messages(self):
        total_discordant_msgs = 0
        for a in self.agents:
            total_discordant_msgs += len(a.discordant_msgs)
        return total_discordant_msgs
   
    
    def is_stationary_state(self, G):
        num_clusters = len([G.subgraph(c) for c in nx.weakly_connected_components(G)])
        num_coverged_clusters = 0

        if num_clusters >= 2:
            for C in [G.subgraph(c) for c in nx.weakly_connected_components(G)]:
                _agents = [self.agents[i] for i in list(C.nodes())]
                opinions = np.array([a.opinion for a in _agents])
                opi_diff = np.max(opinions) - np.min(opinions)
                if opi_diff <= self.epsilon:
                    num_coverged_clusters += 1

        if num_coverged_clusters == num_clusters:
            return True
        else:
            return False

        
    def export_csv(self, data_dic, ofname):
        dir_path = os.path.join(self.data_dir, 'data')
        file_path = os.path.join(dir_path, ofname)
        pd.DataFrame(data_dic).to_csv(file_path, compression='xz')
    
    
    def export_gexf(self, t):
        network_dir_path = os.path.join(self.data_dir, 'network_data')
        file_path = os.path.join(network_dir_path, 'G_' + str(t).zfill(7) + '.gexf.bz2')
        cls = [float(a.opinion) for a in self.agents]
        self.social_media.set_node_colors(cls)
        nx.write_gexf(self.social_media.G, file_path)

        
    def final_exports(self, t):
        self.export_csv(self.opinion_data, 'opinions.csv.xz')
        self.export_csv(self.screen_diversity_data, 'screen_diversity.csv.xz')
        self.social_media.message_df.to_csv(os.path.join(self.data_dir + '/data', 'messages.csv.xz'))
        self.export_gexf(t)

        
    def evolve(self, t_max, mu, p, q, rewiring_methods):
        for t in range(t_max):
            #print("t = ", t)
            self.opinion_data.append([a.opinion for a in self.agents])
            self.screen_diversity_data.append([a.screen_diversity for a in self.agents])

            # export network data
            if t % 10000 == 0:
                self.export_gexf(t)

            # select agent i at random
            user_id = np.random.choice(self.num_agents)

            # agent i refleshes its screen and reading it
            screen = self.social_media.show_screen(user_id)
            self.agents[user_id].evaluate_messages(screen)
            self.agents[user_id].screen_diversity = analysis.screen_diversity(screen.content.values, bins=10)

            # social influence (mu)
            unfollow_id = None
            follow_id = None
            self.agents[user_id].update_opinion(mu)

            # rewiring (q)
            if np.random.random() < q:
                unfollow_id, follow_id = self.agents[user_id].decide_to_rewire(self.social_media, rewiring_methods)
                if unfollow_id is not None and follow_id is not None:
                    self.social_media.rewire_users(user_id, unfollow_id, follow_id)

            # post (1-p) or repost (p) a message 
            msg = self.agents[user_id].post_message(t, p)
            self.social_media.update_message_db(t, msg)
       
            # finalize and export data            
            if self.is_stationary_state(self.social_media.G):
                self.final_exports(t)
                break
            elif t >= t_max - 1:
                self.final_exports(t)
                break


                
if __name__ == '__main__':
    # parameters
    n_agents = 100
    n_links = 400
    sns_seed = 1
    l = 10 # screen size
    t_max = 100000 # max steps
    epsilon = 0.4 # bounded confidence parameter
    mu = 0.5 # social influence strength
    p = 0.5 # repost rate
    q = 0.5 # rewiring rate
    following_methods = ['Random', 'Repost', 'Recommendation']
    
    # simulation
    now_str = pd.datetime.now().strftime('%Y%m%d%H%M%S')
    data_root_dir = os.path.join('data_' + ''.join(now_str))
    s = pd.datetime.now()
    print(s)
    d = EchoChamberDynamics(n_agents, n_links, epsilon, sns_seed, l, data_root_dir)
    d.evolve(t_max, mu, p, q, following_methods)
    e = pd.datetime.now()
    print(e)
    print(e - s)