# %%
import os

os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("tiny-stories-2L-33M")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab

# %%
# data = load_dataset("skeskinen/TinyStories-GPT4")
# # %%
# data_path = Path("/workspace/Tiny-Stories-SAEs/TinyStoriesV2-GPT4-valid.txt")
# data = open(data_path, "r").readlines()
# print(data[:3])
# # %%
# for i in data:
#     print(i)
# %%
import requests
def load_tinystories_validation_prompts(data_path = "/workspace/data/tinystories", file_name = "validation.parquet") -> list[str]:
    Path(data_path).mkdir(parents=True, exist_ok=True)
    file_path = f'{data_path}/{file_name}'
    if not os.path.isfile(file_path):
        print("Downloading TinyStories validation prompts")
        validation_data_path = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/validation-00000-of-00001-869c898b519ad725.parquet"
        response = requests.get(validation_data_path)
        with open(file_path, "wb") as f:
            f.write(response.content)

    # Load Parquet into Pandas DataFrame
    df = pd.read_parquet(file_path, engine='pyarrow')
    prompts = df["text"].tolist()
    # Fix encoding issue for special characters like em-dash
    # prompts = [prompt.encode("windows-1252").decode("utf-8", errors="ignore") for prompt in prompts]
    logging.info(f"Loaded {len(prompts)} TinyStories validation prompts")
    return prompts
# %%
prompts = load_tinystories_validation_prompts()
prompts[0]
# %%
model(prompts[0], return_type="loss")
# %%
tokens = model.to_tokens(prompts[0])
logits = model(tokens)
import circuitsvis as cv
html = cv.logits.token_log_probs(tokens, logits.log_softmax(dim=-1), model.to_string)
type(html)
# %%
story = """Once upon a time, there was a little bear named Timmy. Timmy had a big, yellow bus toy that he loved very much. Every day, he played with his bus, making it drive all around his room.

One day, Timmy's mom said, "Timmy, let's go on a real bus ride today!" Timmy was so excited. He took his toy bus with him.

On the big, real bus, Timmy found a seat. But the seat was hard and made him feel uncomfortable. He squirmed and couldn't sit still. Looking at his toy bus, he remembered how his toy bus had soft seats. "Mom, I wish this seat was soft like my toy bus," Timmy said.

"Let's try to relax," said his mom. "Close your eyes and think of riding in your soft toy bus." Timmy closed his eyes and imagined. He felt better and smiled.

As the bus moved, Timmy started to feel happy. He was on a real bus, just like his toy. When they got off, Timmy said, "Mom, the real bus was fun, even if the seats were not soft."""
model(story, return_type="loss")
# %%
model.generate("Once upon a time", max_new_tokens=200)

# %%
import transformer_lens
from transformer_lens import HookedTransformer, utils
import torch
import numpy as np
import gradio as gr
import pprint
import json
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from IPython.display import HTML
from functools import partial
import tqdm.notebook as tqdm
import plotly.express as px
import pandas as pd

# %%
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to(cfg["device"])
    
    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
    
    def get_version(self):
        version_list = [int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)]
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    # def save(self):
    #     version = self.get_version()
    #     torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
    #     with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
    #         json.dump(cfg, f)
    #     print("Saved as version", version)
    
    # @classmethod
    # def load(cls, version):
    #     cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
    #     pprint.pprint(cfg)
    #     self = cls(cfg=cfg)
    #     self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
    #     return self

    # @classmethod
    # def load_from_hf(cls, version, device_override=None):
    #     """
    #     Loads the saved autoencoder from HuggingFace. 
        
    #     Version is expected to be an int, or "run1" or "run2"

    #     version 25 is the final checkpoint of the first autoencoder run,
    #     version 47 is the final checkpoint of the second autoencoder run.
    #     """
    #     if version=="run1":
    #         version = 25
    #     elif version=="run2":
    #         version = 47
        
    #     cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
    #     if device_override is not None:
    #         cfg["device"] = device_override

    #     pprint.pprint(cfg)
    #     self = cls(cfg=cfg)
    #     self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
    #     return self

# %%
sae0_sd = torch.load("/workspace/Tiny-Stories-SAEs/saes/2L/47_polar_thunder.pt")
sae1_sd = torch.load("/workspace/Tiny-Stories-SAEs/saes/2L/185_upbeat_field.pt")
# %%
cfg = {
    "dict_size": d_mlp * 4,
    "l1_coeff": -1.,
    "enc_dtype": "fp32",
    "seed": 45,
    "act_size": d_mlp,
    "device": "cuda",
}
sae0 = AutoEncoder(cfg)
sae0.load_state_dict(sae0_sd)
sae1 = AutoEncoder(cfg)
sae1.load_state_dict(sae1_sd)
# %%
batch_size = 384
seq_len = 150
tokens = model.to_tokens(prompts[:batch_size*2], prepend_bos=False)
short_stories = tokens[:, seq_len] == model.tokenizer.pad_token_id
tokens = tokens[~short_stories][:batch_size, :seq_len]
print(tokens.shape)

imshow(tokens == model.tokenizer.pad_token_id)
# %%

loss_ex, cache_ex = model.run_with_cache(tokens, return_type="loss", names_filter = lambda x: x.endswith("hook_post"))
print(loss_ex)
# %%
_, mlp0_recons_ex, sae0_hidden_ex, _, _ = sae0(cache_ex["post", 0])
_, mlp1_recons_ex, sae1_hidden_ex, _, _ = sae1(cache_ex["post", 1])
(sae0_hidden_ex>0).sum(-1).float().mean(), (sae1_hidden_ex>0).sum(-1).float().mean()
# %%
cos_sim_ex_0 = nutils.cos(mlp0_recons_ex, cache_ex["post", 0])
print("Cos sim mean", cos_sim_ex_0.mean(), "median", cos_sim_ex_0.median())
cos_sim_ex_1 = nutils.cos(mlp1_recons_ex, cache_ex["post", 1])
print("Cos sim mean", cos_sim_ex_1.mean(), "median", cos_sim_ex_1.median())
# %%
print("Norm of difference / mlp act norm", ((mlp0_recons_ex - cache_ex["post", 0]).norm(dim=-1) / cache_ex["post", 0].norm(dim=-1)).median())
print("Norm of difference / mlp act norm", ((mlp1_recons_ex - cache_ex["post", 1]).norm(dim=-1) / cache_ex["post", 1].norm(dim=-1)).median())
# %%
def replace_mlp_post_hook(mlp_post, hook, new_mlp_post):
    mlp_post[:] = new_mlp_post
    return mlp_post
zero_abl_mlp0_loss = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post_hook, new_mlp_post=0.))], return_type="loss")
repl_mlp0_loss = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post_hook, new_mlp_post=mlp0_recons_ex))], return_type="loss")
zero_abl_mlp1_loss = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", 1), partial(replace_mlp_post_hook, new_mlp_post=0.))], return_type="loss")
repl_mlp1_loss = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", 1), partial(replace_mlp_post_hook, new_mlp_post=mlp1_recons_ex))], return_type="loss")
repl_both_loss = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post_hook, new_mlp_post=mlp0_recons_ex)), (utils.get_act_name("post", 1), partial(replace_mlp_post_hook, new_mlp_post=mlp1_recons_ex))], return_type="loss")

print("Zero abl mlp0 loss", zero_abl_mlp0_loss)
print("Zero abl mlp1 loss", zero_abl_mlp1_loss)
print("Repl mlp0 loss", repl_mlp0_loss)
print("Repl mlp1 loss", repl_mlp1_loss)
print("Repl both loss", repl_both_loss)
print("Baseline loss", loss_ex)
print("Layer 0 loss recovered", (zero_abl_mlp0_loss - repl_mlp0_loss) / (zero_abl_mlp0_loss - loss_ex))
print("Layer 1 loss recovered", (zero_abl_mlp1_loss - repl_mlp1_loss) / (zero_abl_mlp1_loss - loss_ex))
# %%
repl_mlp0_logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post_hook, new_mlp_post=mlp0_recons_ex))], return_type="logits")
repl_mlp0_clps = model.loss_fn(repl_mlp0_logits, tokens, True)
repl_mlp1_logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", 1), partial(replace_mlp_post_hook, new_mlp_post=mlp1_recons_ex))], return_type="logits")
repl_mlp1_clps = model.loss_fn(repl_mlp1_logits, tokens, True)
logits_ex = model(tokens)
clps_ex = model.loss_fn(logits_ex, tokens, True)
scatter(repl_mlp0_clps.flatten(), clps_ex.flatten(), xaxis="Replace MLP0", yaxis="Baseline", include_diag=True, opacity=0.2)
scatter(repl_mlp1_clps.flatten(), clps_ex.flatten(), xaxis="Replace MLP1", yaxis="Baseline", include_diag=True, opacity=0.2)
# %%
logits_ex = model(tokens)
clps_ex = model.loss_fn(logits_ex, tokens, True)
token_df = nutils.make_token_df(tokens)
token_df = token_df[token_df.pos < token_df.pos.max()]
token_df["clp"] = to_numpy(clps_ex.flatten())
token_df.head(5)
# %%
histogram(((sae0_hidden_ex>0).float().mean([0, 1])+10**-7).log10(), histnorm="percent")
histogram(((sae1_hidden_ex>0).float().mean([0, 1])+10**-7).log10(), histnorm="percent")
# %%

batch_size = 384
seq_len = 150
tokens = model.to_tokens(prompts[:batch_size*2], prepend_bos=True)
short_stories = tokens[:, seq_len] == model.tokenizer.pad_token_id
tokens = tokens[~short_stories][:batch_size, :seq_len]
print(tokens.shape)

imshow(tokens == model.tokenizer.pad_token_id)
# %%

loss_ex, cache_ex = model.run_with_cache(tokens, return_type="loss", names_filter = lambda x: x.endswith("hook_post"))
print(loss_ex)
# %%
_, mlp0_recons_ex, sae0_hidden_ex, _, _ = sae0(cache_ex["post", 0])
_, mlp1_recons_ex, sae1_hidden_ex, _, _ = sae1(cache_ex["post", 1])
(sae0_hidden_ex>0).sum(-1).float().mean(), (sae1_hidden_ex>0).sum(-1).float().mean()
# %%
tokens_no_bos = model.to_tokens(prompts[:batch_size*2], prepend_bos=False)
short_stories = tokens_no_bos[:, seq_len] == model.tokenizer.pad_token_id
tokens_no_bos = tokens_no_bos[~short_stories][:batch_size, :seq_len]
print(tokens_no_bos.shape)

imshow(tokens_no_bos == model.tokenizer.pad_token_id)
# %%

loss_ex_no_bos, cache_ex_no_bos = model.run_with_cache(tokens_no_bos, return_type="loss", names_filter = lambda x: x.endswith("hook_post"))
print(loss_ex_no_bos)
# %%
_, mlp0_recons_ex_no_bos, sae0_hidden_ex_no_bos, _, _ = sae0(cache_ex_no_bos["post", 0])
_, mlp1_recons_ex_no_bos, sae1_hidden_ex_no_bos, _, _ = sae1(cache_ex_no_bos["post", 1])
(sae0_hidden_ex_no_bos>0).sum(-1).float().mean(), (sae1_hidden_ex_no_bos>0).sum(-1).float().mean()

# %%
histogram(((sae0_hidden_ex>0).float().mean([0, 1])+10**-7).log10(), histnorm="percent")
histogram(((sae1_hidden_ex>0).float().mean([0, 1])+10**-7).log10(), histnorm="percent")

# %%
feature_df0 = pd.DataFrame({"feature": np.arange(d_mlp*4), "freq": to_numpy((sae0_hidden_ex>0).float().mean([0, 1]))})
feature_df0
# %%
feature_df0["freq_with_bos"] = to_numpy((sae0_hidden_ex>0).float().mean([0, 1]))
feature_df0
# %%
temp_df = copy.deepcopy(token_df)
temp_df["f16381"] = to_numpy(sae0_hidden_ex[:, :-1, 16381].flatten())
temp_df["f16381_no_bos"] = to_numpy(sae0_hidden_ex_no_bos[:, :-1, 16381].flatten())
temp_df.sort_values("f16381_no_bos", ascending=False).query("f16381_no_bos > 0.")

# %%
# START HERE
random.seed(123)
random.shuffle(prompts)
batch_size = 384
seq_len = 150
tokens = model.to_tokens(prompts[:batch_size*2], prepend_bos=True)
short_stories = tokens[:, seq_len] == model.tokenizer.pad_token_id
tokens = tokens[~short_stories][:batch_size, :seq_len]
print(tokens.shape)
print(model.to_string(tokens[0, :30]))

loss_ex, cache_ex = model.run_with_cache(tokens, return_type="loss", names_filter = lambda x: x.endswith("hook_post"))
print(loss_ex)
_, mlp0_recons_ex, sae0_hidden_ex, _, _ = sae0(cache_ex["post", 0])
_, mlp1_recons_ex, sae1_hidden_ex, _, _ = sae1(cache_ex["post", 1])
print("L0 norm", (sae0_hidden_ex>0).sum(-1).float().mean(), (sae1_hidden_ex>0).sum(-1).float().mean())

# %%
token_df = nutils.make_token_df(tokens, len_prefix=10)
token_df = token_df.query(f"4<pos<{seq_len-1}")
token_df
token_df_base = copy.deepcopy(token_df)

# %%
feature_df0 = pd.DataFrame({"feature": np.arange(d_mlp*4), "freq": to_numpy((sae0_hidden_ex>0).float().mean([0, 1]))})
feature_df0["layer"] = 0
feature_df1 = pd.DataFrame({"feature": np.arange(d_mlp*4), "freq": to_numpy((sae1_hidden_ex>0).float().mean([0, 1]))})
feature_df1["layer"] = 1
feature_df = pd.concat([feature_df0, feature_df1])
feature_df_base = copy.deepcopy(feature_df)
feature_df["freq_log"] = np.log10(feature_df["freq"]+10**-7)
px.histogram(feature_df, x="freq_log", histnorm="percent", title="Freq log", color="layer",barmode="overlay").show()
feature_df["is_dead"] = feature_df["freq"]>0.
# %%
logits_soup, cache_soup = model.run_with_cache(text_soup)
_, mlp0_recons_soup, sae0_hidden_soup, _, _ = sae0(cache_soup["post", 0])
_, mlp1_recons_soup, sae1_hidden_soup, _, _ = sae1(cache_soup["post", 1])
text_soup = """Jane takes a spoonful of soup, but then she makes a face. The soup is very"""
utils.test_prompt(text_soup, "bitter", model)
# %%
clean_prompt = "Jane takes a spoonful of soup, and then she makes a face. The soup is very"
clean_answer = " bitter"
clean_token = model.to_single_token(clean_answer)
corr_prompt = "Jane takes a spoonful of soup, and then she starts to smile. The soup is very"
corr_answer = " tasty"
corr_token = model.to_single_token(corr_answer)
clean_logits, clean_cache = model.run_with_cache(clean_prompt)
corr_logits, corr_cache = model.run_with_cache(corr_prompt)
print(clean_logits.softmax(dim=-1)[0, -1, clean_token])
print(clean_logits.softmax(dim=-1)[0, -1, corr_token])
print(corr_logits.softmax(dim=-1)[0, -1, corr_token])
print(corr_logits.softmax(dim=-1)[0, -1, clean_token])
# %%
clean_baseline = clean_logits[0, -1, clean_token] - clean_logits[0, -1, corr_token]
corr_baseline = corr_logits[0, -1, clean_token] - corr_logits[0, -1, corr_token]
def metric(logits, normalize=True):
    if len(logits.shape)==3:
        logits = logits[0, -1]
    logit_diff = (logits[clean_token] - logits[corr_token])
    if normalize:
        return ((logit_diff - corr_baseline) / (clean_baseline - corr_baseline)).item()
    else:
        return logit_diff.item()

metric(clean_logits), metric(corr_logits)
# %%
unembed_vec = model.W_U[:, clean_token] - model.W_U[:, corr_token]
clean_resid_stack, resid_labels = clean_cache.decompose_resid(apply_ln=True, pos_slice=-1, return_labels=True)
corr_resid_stack, resid_labels = corr_cache.decompose_resid(apply_ln=True, pos_slice=-1, return_labels=True)
line([clean_resid_stack @ unembed_vec, corr_resid_stack @ unembed_vec], x=resid_labels, line_labels=["clean", "corr"])
# %%
ZERO = 0
ONE = 1
clean_reconstr_mlp0, clean_hidden0 = sae0(clean_cache["post", 0])[1:3]
clean_reconstr_mlp1, clean_hidden1 = sae1(clean_cache["post", 1])[1:3]
corr_reconstr_mlp0, corr_hidden0 = sae0(corr_cache["post", 0])[1:3]
corr_reconstr_mlp1, corr_hidden1 = sae1(corr_cache["post", 1])[1:3]
def replace_mlp_post_hook(mlp_post, hook, new_mlp_post):
    mlp_post[:] = new_mlp_post
    return mlp_post
new_clean_logits_both = model.run_with_hooks(clean_prompt, fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post_hook, new_mlp_post=clean_reconstr_mlp0)), (utils.get_act_name("post", 1), partial(replace_mlp_post_hook, new_mlp_post=clean_reconstr_mlp1))])
new_clean_logits_repl0 = model.run_with_hooks(clean_prompt, fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post_hook, new_mlp_post=clean_reconstr_mlp0))])
new_clean_logits_repl1 = model.run_with_hooks(clean_prompt, fwd_hooks=[(utils.get_act_name("post", 1), partial(replace_mlp_post_hook, new_mlp_post=clean_reconstr_mlp1))])
new_corr_logits_both = model.run_with_hooks(corr_prompt, fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post_hook, new_mlp_post=corr_reconstr_mlp0)), (utils.get_act_name("post", 1), partial(replace_mlp_post_hook, new_mlp_post=corr_reconstr_mlp1))])
new_corr_logits_repl0 = model.run_with_hooks(corr_prompt, fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post_hook, new_mlp_post=corr_reconstr_mlp0))])
new_corr_logits_repl1 = model.run_with_hooks(corr_prompt, fwd_hooks=[(utils.get_act_name("post", 1), partial(replace_mlp_post_hook, new_mlp_post=corr_reconstr_mlp1))])

print("Clean Replaced both", metric(new_clean_logits_both), metric(new_clean_logits_both, False))
print("Clean replaced MLP0", metric(new_clean_logits_repl0), metric(new_clean_logits_repl0, False))
print("Clean replaced MLP1", metric(new_clean_logits_repl1), metric(new_clean_logits_repl1, False))
print("corr Replaced both", metric(new_corr_logits_both), metric(new_corr_logits_both, False))
print("corr replaced MLP0", metric(new_corr_logits_repl0), metric(new_corr_logits_repl0, False))
print("corr replaced MLP1", metric(new_corr_logits_repl1), metric(new_corr_logits_repl1, False))
# %%
line([
    (clean_hidden0>0)[0].sum(-1),
    (clean_hidden1>0)[0].sum(-1),
    (corr_hidden0>0)[0].sum(-1),
    (corr_hidden1>0)[0].sum(-1),
    ],
    line_labels=["clean MLP0", "clean MLP1", "corr MLP0", "corr MLP1"], x=nutils.process_tokens_index(clean_prompt))
# %%
clean_reconstr_mlp = torch.stack([clean_reconstr_mlp0[0], clean_reconstr_mlp1[0]])
corr_reconstr_mlp = torch.stack([corr_reconstr_mlp0[0], corr_reconstr_mlp1[0]])
clean_hidden = torch.stack([clean_hidden0[0], clean_hidden1[0]])
corr_hidden = torch.stack([corr_hidden0[0], corr_hidden1[0]])


# %%
feature_df = copy.deepcopy(feature_df_base)
W_dec = torch.stack([sae0.W_dec, sae1.W_dec], dim=0)
print(W_dec.shape)
feature_df["wdla"] = to_numpy((W_dec @ model.W_out @ unembed_vec).flatten() / clean_cache["scale"][0, -1, 0])
feature_df["final_act_clean"] = to_numpy(clean_hidden[:, -1, :].flatten())
feature_df["final_act_corr"] = to_numpy(corr_hidden[:, -1, :].flatten())
feature_df["final_act_diff"] = feature_df["final_act_clean"] - feature_df["final_act_corr"]
feature_df["dla"] = feature_df["final_act_diff"] * feature_df["wdla"]
feature_df["alive_on_final"] = (feature_df["final_act_clean"] > 0) | (feature_df["final_act_corr"] > 0) 
feature_df
# %%
nutils.show_df(feature_df.query("alive_on_final").sort_values("dla", ascending=False))
# %%
top_features = feature_df.query("alive_on_final").sort_values("dla", ascending=False).query("layer==1").feature.iloc[:10].values
print(top_features)
# %%
f_id = top_features[0]
token_df = copy.deepcopy(token_df_base)
token_df["act"] = to_numpy(sae1_hidden_ex[:, 5:-1, f_id].flatten())
nutils.show_df(token_df.sort_values("act", ascending=False).query("act > 0"))
# %%
token_df["is_very"] = to_numpy(tokens[:, 5:-1].flatten()==model.to_single_token(" very"))
px.histogram(token_df, x="act", histnorm="percent", title="Freq log", color="is_very",barmode="overlay").show()
token_df["next_token"] = model.to_str_tokens(tokens[:, 6:].flatten())
token_df["act_is_firing"] = token_df["act"]>0
display(token_df.query("is_very").groupby("next_token")["act"].median().sort_values(ascending=False).head(30))
display((token_df.query("is_very").groupby("next_token")["act_is_firing"]).mean().sort_values(ascending=False).head(30))
px.box(token_df.query("is_very"), x="next_token", y="act")
# %%
token_df.query("is_very & next_token==' pretty' & act>0")
# %%
nutils.show_df(nutils.create_vocab_df(sae1.W_dec[f_id, :] @ model.W_out[1] @ model.W_U).head(50))
# %%
f_id = top_features[0]
print(feature_df.query("feature==@f_id & layer==1").iloc[0])
token_df = copy.deepcopy(token_df_base)
token_df["next_token"] = model.to_str_tokens(tokens[:, 6:].flatten())
token_df["act"] = to_numpy(sae1_hidden_ex[:, 5:-1, f_id].flatten())
nutils.show_df(token_df.sort_values("act", ascending=False).query("act > 0"))
nutils.show_df(nutils.create_vocab_df(sae1.W_dec[f_id, :] @ model.W_out[1] @ model.W_U).head(30))
nutils.show_df(nutils.create_vocab_df(sae1.W_dec[f_id, :] @ model.W_out[1] @ model.W_U).tail(20))

# %%
mlp_post_store = [None]
def store_mlp_post_hook(mlp_post, hook):
    mlp_post_store[0] = (mlp_post[0, -1])
    return 
model.reset_hooks(including_permanent=False)
model.blocks[1].mlp.hook_post.add_perma_hook(store_mlp_post_hook)
# %%
f_id = 8651
layer = 1
def sae_metric(mlp_post, normalize=True):
    if type(mlp_post)==ActivationCache:
        mlp_post = mlp_post["post", 1][0, -1]
    feature_act = ((mlp_post - sae1.b_dec) @ sae1.W_enc[:, f_id] + sae1.b_enc[f_id]).item()
    if normalize:
        return (feature_act - corr_sae_baseline) / (clean_sae_baseline - corr_sae_baseline)
    else:
        return feature_act
clean_sae_baseline = sae_metric(clean_cache, normalize=False)
corr_sae_baseline = sae_metric(corr_cache, normalize=False)
print(f"{clean_sae_baseline=}, {corr_sae_baseline=}")

# %%
def replace_z(z, hook, new_z, head):
    z[:, :, head, :] = new_z
    return z
records = []
for layer in range(n_layers):
    for head in range(n_heads):
        noised_logits = model.run_with_hooks(clean_prompt, fwd_hooks=[(utils.get_act_name("z", layer), partial(replace_z, new_z=corr_cache["z", layer][:, :, head, :], head=head))])
        noised_mlp_post = mlp_post_store[0]
        denoised_logits = model.run_with_hooks(corr_prompt, fwd_hooks=[(utils.get_act_name("z", layer), partial(replace_z, new_z=clean_cache["z", layer][:, :, head, :], head=head))])
        denoised_mlp_post = mlp_post_store[0]
        records.append({
            "layer": layer,
            "head": head,
            'label': f"L{layer}H{head}",
            "sae_metric_n": sae_metric(noised_mlp_post),
            "logit_diff_n": metric(noised_logits),
            "sae_metric_dn": sae_metric(denoised_mlp_post),
            "logit_diff_dn": metric(denoised_logits),
            "site": "z",
        })
        
df = pd.DataFrame(records)
px.line(df, x="label", y=["sae_metric_n", "sae_metric_dn", "logit_diff_n", "logit_diff_dn"]).show()
# %%
df["score"] = 2 + (df["sae_metric_dn"] - df["sae_metric_n"]) + (df["logit_diff_dn"] - df["logit_diff_n"])
df = df.sort_values("score", ascending=False)
nutils.show_df(df)
# %%
layers = []
heads = []
for i in range(10):
    layers.append(df.layer.iloc[i])
    heads.append(df["head"].iloc[i])
labels = df.label.iloc[:10].values
imshow(clean_cache.stack_activation("pattern")[layers, 0, heads, -1], x=nutils.process_tokens_index(clean_prompt), title="Clean Attn", y=labels)
imshow(corr_cache.stack_activation("pattern")[layers, 0, heads, -1], x=nutils.process_tokens_index(corr_prompt), title="Corr Attn", y=labels)
# %%
def replace_z(z, hook, new_z, head):
    z[:, 14, head, :] = new_z
    return z
records = []
for layer in range(n_layers):
    for head in range(n_heads):
        noised_logits = model.run_with_hooks(clean_prompt, fwd_hooks=[(utils.get_act_name("z", layer), partial(replace_z, new_z=corr_cache["z", layer][:, 14, head, :], head=head))])
        noised_mlp_post = mlp_post_store[0]
        denoised_logits = model.run_with_hooks(corr_prompt, fwd_hooks=[(utils.get_act_name("z", layer), partial(replace_z, new_z=clean_cache["z", layer][:, 14, head, :], head=head))])
        denoised_mlp_post = mlp_post_store[0]
        records.append({
            "layer": layer,
            "head": head,
            'label': f"L{layer}H{head}",
            "sae_metric_n": sae_metric(noised_mlp_post),
            "logit_diff_n": metric(noised_logits),
            "sae_metric_dn": sae_metric(denoised_mlp_post),
            "logit_diff_dn": metric(denoised_logits),
            "site": "z",
        })
        
df = pd.DataFrame(records)
px.line(df, x="label", y=["sae_metric_n", "sae_metric_dn", "logit_diff_n", "logit_diff_dn"]).show()
# %%
