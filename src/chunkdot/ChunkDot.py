{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 190,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import time as time\n",
    "import contextlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Chunk:\n",
    "    def __init__(self, X,Y=None):\n",
    "        self.X = X\n",
    "        self.Y = Y\n",
    "\n",
    "        if self.Y == None:\n",
    "            self.Y = self.X\n",
    "\n",
    "        self.device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "    def sim_matrix(self, a, b, eps=1e-8):\n",
    "        \"\"\"\n",
    "        Compute the cosine similarity between two matrices of vectors\n",
    "        :param a: matrix of vectors (n x d)\n",
    "        :param b: matrix of vectors (m x d)\n",
    "        :param eps: added eps for numerical stability\n",
    "        :return: scalar product between each vector of a and each vector of b (n x m)\n",
    "        \"\"\"\n",
    "        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]\n",
    "        a_norm = a / torch.clamp(a_n, min=eps)\n",
    "        b_norm = b / torch.clamp(b_n, min=eps)\n",
    "        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))\n",
    "        return sim_mt\n",
    "    \n",
    "    def compute_sim_matrix(self, keep_n=10, chunk_size=100, verbose=True):\n",
    "        \"\"\"\n",
    "        Compute the similarity matrix between X and Y and return the indices of the top-n elements as well as the distances\n",
    "        Args:\n",
    "        keep_n: number of elements to keep\n",
    "        chunk_size: size of the chunks to split the data. This is useful to avoid memory issues\n",
    "        \"\"\"\n",
    "        assert keep_n <= chunk_size, \"keep_n should be less than or equal to chunk_size\"\n",
    "        assert keep_n <= self.Y.shape[0], \"keep_n should be less or equal to the number of elements in Y\"\n",
    "        if self.device == \"cuda\":\n",
    "            torch.cuda.empty_cache()\n",
    "        indices = torch.zeros(self.X.shape[0], keep_n)\n",
    "        distances = torch.zeros(self.X.shape[0], keep_n)\n",
    "\n",
    "        splits_X = self.X.split(chunk_size,dim=0)\n",
    "        split_lenght_X = [i.shape[0] for i in splits_X]\n",
    "\n",
    "        splits_Y = self.Y.split(chunk_size,dim=0)\n",
    "        split_lenghts_Y = [i.shape[0] for i in splits_Y]\n",
    "\n",
    "        print(f\"Number of chunks for X: {len(splits_X)}\")\n",
    "        print(f\"Number of chunks for Y: {len(splits_Y)}\")\n",
    "\n",
    "        start = time.time()\n",
    "\n",
    "        for k,i in enumerate(splits_X):\n",
    "            top_n_all_fused = []\n",
    "            top_n_all_fused_values = []\n",
    "            y_dim = i.shape[0]\n",
    "            for l,j in enumerate(splits_Y):\n",
    "                for l,j in enumerate(splits_Y):\n",
    "                    with contextlib.ExitStack() as stack:\n",
    "                        if self.device == \"cuda\":\n",
    "                            stack.enter_context(torch.cuda.amp.autocast())\n",
    "                        inter = self.sim_matrix(i.to(self.device),j.to(self.device))\n",
    "                        inter, top_n = torch.topk(inter, k=keep_n, dim=1)\n",
    "                        top_n_all_fused_values.append(inter)\n",
    "                        top_n_all_fused.append(top_n+sum(split_lenghts_Y[:l]))\n",
    "\n",
    "                if verbose == True:\n",
    "                    print(f\"Processing of chunk {k+1}/{len(splits_X)} with chunk {l+1}/{len(splits_Y)} done in {time.time()-start:2.3f}s\")\n",
    "\n",
    "            top_n_all_fused = torch.cat(top_n_all_fused,dim=1)\n",
    "            top_n_all_fused_values = torch.cat(top_n_all_fused_values,dim=1)\n",
    "\n",
    "            with contextlib.ExitStack() as stack:\n",
    "                        if self.device == \"cuda\":\n",
    "                            stack.enter_context(torch.cuda.amp.autocast())\n",
    "                        val, top_n_all_fused_values = torch.topk(top_n_all_fused_values,k=keep_n,dim=1)\n",
    "\n",
    "            comb = torch.cat([a[i].reshape(1,-1) for a,i in zip(top_n_all_fused,top_n_all_fused_values)],dim=0)\n",
    "\n",
    "            indices[sum(split_lenght_X[:k]):sum(split_lenght_X[:k])+y_dim] = comb.cpu()\n",
    "            distances[sum(split_lenght_X[:k]):sum(split_lenght_X[:k])+y_dim] = val.cpu()\n",
    "\n",
    "            del val, comb, top_n_all_fused, top_n_all_fused_values, inter, top_n\n",
    "            if self.device == \"cuda\":\n",
    "                torch.cuda.empty_cache()\n",
    "        return indices, distances\n",
    "    \n",
    "    def get_chunk_size(self):\n",
    "        pass\n",
    "\n",
    "    def verbose(self, *args):\n",
    "        pass\n",
    "        \n",
    "        \n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of chunks for X: 5\n",
      "Number of chunks for Y: 10\n",
      "Processing of chunk 1/5 with chunk 10/10 done in 0.070s\n",
      "Processing of chunk 1/5 with chunk 10/10 done in 0.122s\n",
      "Processing of chunk 1/5 with chunk 10/10 done in 0.166s\n",
      "Processing of chunk 1/5 with chunk 10/10 done in 0.203s\n",
      "Processing of chunk 1/5 with chunk 10/10 done in 0.263s\n",
      "Processing of chunk 1/5 with chunk 10/10 done in 0.310s\n",
      "Processing of chunk 1/5 with chunk 10/10 done in 0.346s\n",
      "Processing of chunk 1/5 with chunk 10/10 done in 0.381s\n",
      "Processing of chunk 1/5 with chunk 10/10 done in 0.417s\n",
      "Processing of chunk 1/5 with chunk 10/10 done in 0.455s\n",
      "Processing of chunk 2/5 with chunk 10/10 done in 0.532s\n",
      "Processing of chunk 2/5 with chunk 10/10 done in 0.575s\n",
      "Processing of chunk 2/5 with chunk 10/10 done in 0.611s\n",
      "Processing of chunk 2/5 with chunk 10/10 done in 0.649s\n",
      "Processing of chunk 2/5 with chunk 10/10 done in 0.700s\n",
      "Processing of chunk 2/5 with chunk 10/10 done in 0.740s\n",
      "Processing of chunk 2/5 with chunk 10/10 done in 0.777s\n",
      "Processing of chunk 2/5 with chunk 10/10 done in 0.813s\n",
      "Processing of chunk 2/5 with chunk 10/10 done in 0.851s\n",
      "Processing of chunk 2/5 with chunk 10/10 done in 0.892s\n",
      "Processing of chunk 3/5 with chunk 10/10 done in 0.949s\n",
      "Processing of chunk 3/5 with chunk 10/10 done in 0.984s\n",
      "Processing of chunk 3/5 with chunk 10/10 done in 1.020s\n",
      "Processing of chunk 3/5 with chunk 10/10 done in 1.056s\n",
      "Processing of chunk 3/5 with chunk 10/10 done in 1.092s\n",
      "Processing of chunk 3/5 with chunk 10/10 done in 1.124s\n",
      "Processing of chunk 3/5 with chunk 10/10 done in 1.159s\n",
      "Processing of chunk 3/5 with chunk 10/10 done in 1.195s\n",
      "Processing of chunk 3/5 with chunk 10/10 done in 1.250s\n",
      "Processing of chunk 3/5 with chunk 10/10 done in 1.300s\n",
      "Processing of chunk 4/5 with chunk 10/10 done in 1.386s\n",
      "Processing of chunk 4/5 with chunk 10/10 done in 1.438s\n",
      "Processing of chunk 4/5 with chunk 10/10 done in 1.487s\n",
      "Processing of chunk 4/5 with chunk 10/10 done in 1.530s\n",
      "Processing of chunk 4/5 with chunk 10/10 done in 1.570s\n",
      "Processing of chunk 4/5 with chunk 10/10 done in 1.612s\n",
      "Processing of chunk 4/5 with chunk 10/10 done in 1.663s\n",
      "Processing of chunk 4/5 with chunk 10/10 done in 1.709s\n",
      "Processing of chunk 4/5 with chunk 10/10 done in 1.760s\n",
      "Processing of chunk 4/5 with chunk 10/10 done in 1.914s\n",
      "Processing of chunk 5/5 with chunk 10/10 done in 2.008s\n",
      "Processing of chunk 5/5 with chunk 10/10 done in 2.158s\n",
      "Processing of chunk 5/5 with chunk 10/10 done in 2.224s\n",
      "Processing of chunk 5/5 with chunk 10/10 done in 2.291s\n",
      "Processing of chunk 5/5 with chunk 10/10 done in 2.355s\n",
      "Processing of chunk 5/5 with chunk 10/10 done in 2.408s\n",
      "Processing of chunk 5/5 with chunk 10/10 done in 2.470s\n",
      "Processing of chunk 5/5 with chunk 10/10 done in 2.523s\n",
      "Processing of chunk 5/5 with chunk 10/10 done in 2.555s\n",
      "Processing of chunk 5/5 with chunk 10/10 done in 2.588s\n"
     ]
    }
   ],
   "source": [
    "a = torch.randn(5000,32)\n",
    "b = torch.randn(10000,32)\n",
    "c = Chunk(a,b)\n",
    "k = c.compute_sim_matrix(keep_n=30, chunk_size=1000, verbose=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/js/hxmnpkdx4fvcpqktc9g8j9p80000gn/T/ipykernel_28140/2498001530.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0ma\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmatrix\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ma\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/numba/core/serialize.py\u001b[0m in \u001b[0;36m_numba_unpickle\u001b[0;34m(address, bytedata, hashed)\u001b[0m\n\u001b[1;32m     28\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     29\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 30\u001b[0;31m \u001b[0;32mdef\u001b[0m \u001b[0m_numba_unpickle\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0maddress\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbytedata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mhashed\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     31\u001b[0m     \"\"\"Used by `numba_unpickle` from _helperlib.c\n\u001b[1;32m     32\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import numba as nb\n",
    "\n",
    "@nb.njit\n",
    "def matrix(a,b):\n",
    "    res = np.zeros((a.shape[0], b.shape[1]))\n",
    "    for i in range(b.shape[1]):\n",
    "        for j in range(a.shape[0]):\n",
    "            for k in range(a.shape[1]):\n",
    "                res[j,i] += a[j,k]*b[k,i]\n",
    "    return res\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = np.random.randn(10000,4000)\n",
    "b = np.random.randn(4000,1000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "b = np.matmul(a,b)\n",
    "b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = matrix(a,b)\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
