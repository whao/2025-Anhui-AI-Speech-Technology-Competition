import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel



ANSWER: dict = {
    'b8df9c6e1e0671400c88bf76ea9b01ee5198f9e077a2b6d05d62ee5eb1926d71': 'fear',
    'c9ae526d0cad6a7a992d515a28d23992bd08052e4067f52e51f763864281b215': 'fear',
    'd6d1628785d457c0e9b581ce65e650f732e06523dda07d58083d2bbd0959a2f5': 'fear',
    '06b0bfc5df3c191e00a255fb09dc3c138ad242be312627d19552be534f0db957': 'fear',
    '3d84b486ec8ef356f0fedc1bc7c1682d6a8b6c505ef2c10efe52a35ff4f7bec5': 'fear',
    '529fe47c444f0d09d3666b44c1d670e27bac39df7e39b7ee0110252dde3888f2': 'fear',
    'e31741f705fee11a233e759af986e4878d0c07a53c8596b86c8843c9b0def45e': 'fear',
    'ec843200bd063e27a6826611e5da281c28c663fa807de8ffa1ed89fcb58b7666': 'fear',
    '91803fad54309abc8d5891d1099091a857b5ea33250ba51ed03e4cd74562ea54': 'fear',
    '90efe02f47ad34469b430fdf85dfa2fe7eb45408d447fabc0a80428c8e62f161': 'fear',
    '4d594588ca878e05cf804d88c27212d117b6dc48197c8ef88564ceb00bb2fff1': 'sad',
    'c8b3a5b86806b0ef9ecc54cda5632f14e1b392bd9906ee082620ba8ebadc7da9': 'sad',
    '91ee6703651ba441bf64f6556566915a84b4ba679ef3e1d02b271c7359288fee': 'sad',
    'bbe6b21e872dde394f633c713dc009acf7d2fa89e286212e3b7086d91b792d52': 'sad',
    'e343040049e7e799f0e087a53e8262e176c18e9df63c17fe9d36c4d90dec7ae1': 'sad',
    '685092119b8bd62b697eaf04e2bbf8f25ec73538eeac02a854ff32b74c2a3f61': 'sad',
    '1b5cd63fd6738db3a95ac7dbacf962148094b1f7c37a3a0e55af42a3979f38aa': 'sad',
    '056fb0e64a845cd000e56e7dce460d806064de7a44b3e4f593f4126af4612dfa': 'sad',
    'f0c53ca61815b2e12eb39c0e1042ffad6b6489ff0e9b16acfc210d8cf1c1b207': 'sad',
    '64386c9f38edac0929eaec61eb84755e5fbf642df920b5397299b1a931e59124': 'sad',
    'f2ba0187d76338363133c30abde84c231c3939edd0284375e703a5a1993b6f6f': 'surprise',
    'efbfd06cc06bfa75fba85af3cb7e69469d87944f4864039276b9807f6c092181': 'surprise',
    '9625017b7688df5d9dfac71f0a642c1c3e7b4e01c4706f11cb7c3f344573da1f': 'surprise',
    '69ad9fd8a68e3b7ec42c7b7d0bf93c3efb13c66d748f3023d02bddbd57a98bfe': 'surprise',
    '57890963bad9b9ca2d538edb4f9e2fcec397886bc25e20a1b42aff67979f98ee': 'surprise',
    'a91a6da34edb5d3c4b638e837568842bf4c4d0059f7505f85e43f520b1e283ef': 'surprise',
    '3b51df12c63a62a02899e3e66f227882bffcfe485b0ce9d93ccd1ca79b86bb11': 'surprise',
    'c0b9c8f9a62a6084b83f9e15204ee2772a48be9f0c030b2f2d760d5abf22aa52': 'surprise',
    '6b724f984a8e1f4576da7c0c174730941fb0fad824154fa4b1058eaa5803b113': 'surprise',
    '01186ea5bd9e0a6e7915a14d0072408d593f3bffa1a3b56aba6ab508bb16c2f9': 'surprise',
    '724442095b6b328189d72cff58740b1416a8ee0799d55a3e81aefa21778006cb': 'neutral',
    '4a8897b3f6414a54022a7dc86798282fff119ce26aeec2f46306667af8d8e7cb': 'neutral',
    'b7ec2a8ccc2b9b1a156727b013009c2e0d03e25220080daa501f9605c62e5549': 'neutral',
    '313ffac918f8f28ad28524f19a8b5b8e0b44216166828c2c5e0f354179138ac1': 'neutral',
    '0f77af89e30a943eb64c9e1c888f03043519e5c5516be02b78c1bd8619dfc4a9': 'neutral',
    '33eab11c55244374722cf98a7ccb6dbe9860718985bd32209c7909a9f3fe01d0': 'neutral',
    '04da1b7a8652b9266449328b1c8b931bb9f10e60c5793bd99fbdf53920024164': 'neutral',
    '1e86b29c591a686dc3486073b2a7d1a396c1761a1b4379957b49350744fdc052': 'neutral',
    '709f73148c4d869a94b316899b990fa95118783f65b32b4c24b0b9750ebe5086': 'neutral',
    '498f9c327b1f4f8569db43c0ba437c4ffa32125f5fecd61aef697d82023f81e5': 'neutral',
    'a6c418db78ecfd8c74ff699d2cb4f448eb2291104283c69f5808dabc1a33692d': 'angry',
    '5c73defa2ee22f4e4173b4f583d51b42b4aa4bfbb1af3a1fa65f89e80cf51434': 'angry',
    '21d7820ef1df77c743057ab1a2cda474ef13798927148cc6cb8d4aa678577e9f': 'angry',
    '8b9581c9232fe4a45f31b4cea5287647907d19bc1a743d9e8b514eb10b2a74a1': 'angry',
    '5965fa6d783a2bef249e68682322b10b7ec569da74c654c0993687c75a18e10d': 'angry',
    '64c05c9ef7eba192b151ad4ad9be352c1f0cfee0398ebb5d088a3752a4e21220': 'angry',
    '44ca8e83f257ba098274a64e41f4526182bb43838de73f5bf6a93c01eed69754': 'angry',
    '4cccdeeeed197091f1488a01e7f49e348cac3a26764c09939f89905172873162': 'angry',
    '5df924d2afc2548833776a7f881e1b50f78bc58cc89b695dbdc8bfa8935ecb81': 'angry',
    'f99d61c98008c830fc4db1b61ee10b33b30bb96d66109ed71719cd5a1dd5f87c': 'angry',
    '6539e0492498a3b621e91c4cc16dcde03c7f8510be88c4df2589d879a47d0156': 'happy',
    'b5f1487ccbba22f38b429d4fb753680acc6570d3f6c723c0eb6509fe10be351a': 'happy',
    '9777e4e495fda9a0d0eea09eab060ce653634ec45dd2213286cca52584b95d70': 'happy',
    '363e0bea5a22d76f01c0665ef26f8daf22cc3c38fdfc213a318a4752deb7c7b4': 'happy',
    '20195b018a6155e3a612b981bc1ebddadfc00f730c9c98bd88978916b2d2c55c': 'happy',
    'f331137029e95e96b6e18123f9626e2b66b8f5e1d04330af0234b9651cc45c25': 'happy',
    '8273ba9fbf46443129d36f9df410608c1beea6f5edeeaf6f340ccec71dcf5d8c': 'happy',
    'b39e0870be8590889d4192665bd247ddc85f32fc22032274941781d7f94a191a': 'happy',
    '2dcc0cd3c9f87e76b8f3add5f0bdab671c14c456900141f611b7030c6443ef12': 'happy',
    'f9587ad0aef671f2ac947f3a6e648bb2823657ec13f440a1f0a5783e03aa45fd': 'happy',
}

# Create a local server to serve the FastAPI application. Receive the hash:emotion pair request and grade it
# by checking if the hash exists in the ANSWER dictionary.
# Request has to be in the length of 60 pairs of hash:emotion. And grade it once for all. Then return the result.


app = FastAPI()
class EmotionRequest(BaseModel):
    pairs: list[dict[str, str]]  # List of dictionaries with 'hash' and 'emotion' keys

@app.post("/grade_emotion/")
async def grade_emotion(request: EmotionRequest):
    if len(request.pairs) != 60:
        return {"error": "Invalid request length. Must be 60 pairs."}

    results = {}
    for pair in request.pairs:
        hash_value = pair.get("hash")
        emotion = pair.get("emotion")
        if hash_value in ANSWER:
            results[hash_value] = ANSWER[hash_value] == emotion

    # Return the result percentage of correctness
    correct_count = sum(results.values())
    total_count = len(results)
    percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
    return {"correct_count": correct_count, "total_count": total_count, "percentage": percentage}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8060)