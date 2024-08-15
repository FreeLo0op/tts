from tal_frontend.frontend.g2p_pp.g2p_pp_client import TAL_G2PPP_Triton

url = "123.56.235.205:80"
    
client = TAL_G2PPP_Triton(url)

# sentences = ['这道题要我们解决的问题是求一个等腰三角形的顶角的度']
sentences = [
"with tenure, suzie d have all the more pleasure for yachting.",
"n b c glad. why? fox t v jerks quiz pm."]


rhy_res_l, phonemes_res_l = client.infer(sentences)
print(rhy_res_l, phonemes_res_l)
