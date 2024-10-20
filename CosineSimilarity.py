import torch
import time

class Chunk:
    def __init__(self, X,Y=None):
        self.X = X
        self.Y = Y

        if self.Y == None:
            self.Y = self.X

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sim_matrix(self, a, b, eps=1e-8):
        """
        Compute the cosine similarity between two matrices of vectors
        :param a: matrix of vectors (n x d)
        :param b: matrix of vectors (m x d)
        :param eps: added eps for numerical stability
        :return: scalar product between each vector of a and each vector of b (n x m)
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def compute_sim_matrix(self, keep_n=10, chunk_size=100, verbose=True):
        """
        Compute the similarity matrix between X and Y and return the indices of the top-n elements as well as the distances
        Args:
        keep_n: number of elements to keep
        chunk_size: size of the chunks to split the data. This is useful to avoid memory issues
        """
        assert keep_n <= chunk_size, "keep_n should be less than or equal to chunk_size"
        assert keep_n <= self.Y.shape[0], "keep_n should be less or equal to the number of elements in Y"
        if self.device == "cuda":
            torch.cuda.empty_cache()
        indices = torch.zeros(self.X.shape[0], keep_n)
        distances = torch.zeros(self.X.shape[0], keep_n)

        splits_X = self.X.split(chunk_size,dim=0)
        split_lenght_X = [i.shape[0] for i in splits_X]

        splits_Y = self.Y.split(chunk_size,dim=0)
        split_lenghts_Y = [i.shape[0] for i in splits_Y]

        print(f"Number of chunks for X: {len(splits_X)}")
        print(f"Number of chunks for Y: {len(splits_Y)}")

        start = time.time()

        for k,i in enumerate(splits_X):
            top_n_all_fused = []
            top_n_all_fused_values = []
            y_dim = i.shape[0]
            for l,j in enumerate(splits_Y):
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        inter = self.sim_matrix(i.to(self.device),j.to(self.device))
                        inter, top_n = torch.topk(inter, k=keep_n, dim=1)
                        top_n_all_fused_values.append(inter)
                        top_n_all_fused.append(top_n+sum(split_lenghts_Y[:l]))
                else:
                    inter = self.sim_matrix(i,j)
                    inter, top_n = torch.topk(inter, k=keep_n, dim=1)
                    top_n_all_fused_values.append(inter)
                    top_n_all_fused.append(top_n+sum(split_lenghts_Y[:l]))

                if verbose == True:
                    print(f"Processing of chunk {k+1}/{len(splits_X)} with chunk {l+1}/{len(splits_Y)} done in {time.time()-start:2.3f}s")

            top_n_all_fused = torch.cat(top_n_all_fused,dim=1)
            top_n_all_fused_values = torch.cat(top_n_all_fused_values,dim=1)

            if self.device == "cuda":    
                with torch.cuda.amp.autocast():
                    val, top_n_all_fused_values = torch.topk(top_n_all_fused_values,k=keep_n,dim=1)
            else:
                val, top_n_all_fused_values = torch.topk(top_n_all_fused_values,k=keep_n,dim=1)

            comb = torch.cat([a[i].reshape(1,-1) for a,i in zip(top_n_all_fused,top_n_all_fused_values)],dim=0)

            indices[sum(split_lenght_X[:k]):sum(split_lenght_X[:k])+y_dim] = comb.cpu()
            distances[sum(split_lenght_X[:k]):sum(split_lenght_X[:k])+y_dim] = val.cpu()

            del val, comb, top_n_all_fused, top_n_all_fused_values, inter, top_n
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return indices, distances
    
    def get_chunk_size(self):
        pass

    def verbose(self, *args):
        pass