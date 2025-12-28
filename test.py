from fastmcp import FastMCP
import json

mcp = FastMCP("myMCP")

@mcp.tool()
def get() -> str:
    result = {
        "user_idx": 52,
        "item_idx": 78,
        "pred_rating": 0.88
    }
    print(json.dumps(result, ensure_ascii=False))
    # return "hahaha"
    s = json.dumps(result, ensure_ascii=False)
    return "" + s

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8001, path="/mcp")
