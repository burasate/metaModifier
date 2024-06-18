# -*- coding: utf-8 -*-
import sys, os
import maya.cmds as cmds

#https://www.unrealengine.com/marketplace/en-US/product/pose-driver-connect
class metahuman_pose_wrangler:
    def __init__(self):
        self.root_dir = os.path.dirname(os.path.abspath(__file__)) + '/epic_pose_wrangler'
        if not os.path.exists(self.root_dir):
            cmds.warning('can\'t found {}'.format(self.root_dir))
            return
        for i in [self.root_dir]:
            if not i in sys.path:
                print(i)
                sys.path.insert(0, i)
        from epic_pose_wrangler.v2 import main
        self.main = main
        self.rbf_api = self.main.UERBFAPI(view=False)
        print('loaded RBF API (EPIC POSE WRANGLER)')