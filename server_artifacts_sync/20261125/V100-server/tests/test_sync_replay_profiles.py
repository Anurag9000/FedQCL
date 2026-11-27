import random,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.replay import select_memory
class ReplayProfiles(unittest.TestCase):
    def test_caps_and_no_duplication(self):
        pool={0:[(i,0) for i in range(1)],1:[(i,i%3) for i in range(20)]}
        for mode in ("uniform","balanced","rare"):
            for budget in (0,1,2,7,50):
                out=select_memory(pool,budget,mode,random.Random(1))
                self.assertEqual(sum(map(len,out.values())),min(budget,21))
                for k,v in out.items():
                    self.assertEqual(len(v),len(set(v)))
                    self.assertTrue(set(v)<=set(pool[k]))
    def test_class_balance(self):
        out=select_memory({0:[(i,int(i==0)) for i in range(10)]},2,"balanced",random.Random(1))
        self.assertEqual({y for _,y in out[0]},{0,1})
