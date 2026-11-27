import sys
from pathlib import Path
from fractions import Fraction
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.policies import Report
from flcore.sync.advanced import cached_views,select_cached,update_match,combine_scores,peer_filter

def hist(i,a,r=0):
    return Report(i,0,r,(0,1),(a,100-a))
def update(i,v):
    return Report(i,0,0,(0,1),v,kind="update",space="native",metric="cosine",sample_count=20)
class Advanced(unittest.TestCase):
    def test_cache_scope_and_unknown(self):
        views=cached_views([hist(0,50),hist(1,50,4),hist(2,80,6)],range(3),0,5,3)
        self.assertIsNone(views[0]);self.assertIsNone(views[2])
        self.assertEqual(select_cached(range(3),{},(),views,2,5,3),(0,2))
    def test_coverage(self):
        views={0:hist(0,0),1:hist(1,10),2:hist(2,100)}
        self.assertEqual(select_cached(range(3),{},(),views,2,0,5),(0,2))
    def test_matching(self):
        views={0:update(0,(-1.,0.)),1:update(1,(0.,0.)),2:update(2,(1.,0.))}
        self.assertEqual(update_match(range(3),{},(),views,1,1,5),(1,))
    def test_hybrid_unknown(self):
        self.assertIsNone(combine_scores(None,Fraction(0),"intersection"))
        self.assertEqual(combine_scores(Fraction(1,4),Fraction(1,2)),Fraction(1,2))
        views={i:hist(i,50) for i in range(3)}
        self.assertEqual(peer_filter(range(3),{},(),views,{0:None,1:Fraction(0),2:Fraction(0)},Fraction(1,10),None,"exact","distance"),(0,))
