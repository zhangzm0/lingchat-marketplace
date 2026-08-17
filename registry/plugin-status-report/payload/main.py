"""状态汇报插件：汇总当前角色、场景与本地时间。

只调用只读内置工具（status_get_current / status_get_scene / get_current_time），
不发起网络请求，不调用任何写操作工具。
"""


def run(ctx):
    call_tool = ctx["call_tool"]
    status = call_tool("status_get_current", {})
    scene = call_tool("status_get_scene", {})
    time_info = call_tool("get_current_time", {})
    return {
        "ok": True,
        "current_role_id": status.get("current_role_id"),
        "scene": scene.get("name"),
        "scene_description": scene.get("description"),
        "local_time": time_info.get("local_time"),
        "timezone": time_info.get("timezone"),
    }
