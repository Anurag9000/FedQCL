import sys,unittest
from pathlib import Path
from dataclasses import replace
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.policies import Report,peer_select
from flcore.sync.advanced import select_cached,update_match
class Weighted(unittest.TestCase):
    def test_weighted_coverage_changes_choice(self):
        reports={0:Report(0,0,0,(0,1),(100,0)),1:Report(1,0,0,(0,1),(0,100),importance=9.)}
        self.assertEqual(peer_select((0,1),{},(),reports,Fraction(1,10),1,weighted=True),(1,))
        self.assertEqual(select_cached((0,1),{},(),reports,1,0,5,weights={0:1.,1:9.}),(1,))
    def test_weighted_update(self):
        reports={i:Report(i,0,0,(0,),(v,),kind="update",space="native",sample_count=1,metric="cosine",aggregation_weight=w)
                 for i,v,w in ((0,0.,1.),(1,5.,1.),(2,10.,20.))}
        self.assertEqual(update_match((0,1,2),{},(),reports,1,0,5),(1,))
        self.assertEqual(update_match((0,1,2),{},(),reports,1,0,5,weighted=True),(2,))
