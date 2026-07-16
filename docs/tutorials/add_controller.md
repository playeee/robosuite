## 添加第三方控制器

要在 robosuite 中使用第三方控制器，您需要：
1. 创建一个新类，继承 `robosuite/controllers/composite/composite_controller.py` 中的某个复合控制器。
2. 使用装饰器 `@register_composite_controller` 注册该复合控制器。
3. 实现复合特定功能，最终为底层 `part_controller` 提供控制输入。
4. 导入新类，以便通过 `@register_composite_controller` 装饰器将其添加到 robosuite 的 `REGISTERED_COMPOSITE_CONTROLLERS_DICT` 中。
5. 在 json 文件中提供控制器特定配置以及新控制器的 `type`。

对于继承 `WholeBody` 的新复合控制器，您主要需要更新 `joint_action_policy`。

我们在 `robosuite/examples/third_party_controller/` 目录中提供了一个示例，展示如何在 robosuite 中使用第三方 `WholeBodyMinkIK` 复合控制器。您可以运行命令 `python teleop_mink.py` 示例脚本来查看第三方控制器的运行情况。注意：要运行此特定示例，您需要 `pip install mink`。


步骤 1 和 2：

在 `robosuite/examples/third_party_controller/mink_controller.py` 中：

```
@register_composite_controller
class WholeBodyMinkIK(WholeBody):
    name = "WHOLE_BODY_MINK_IK"
```

步骤 3：

在 `robosuite/examples/third_party_controller/mink_controller.py` 中，添加特定于新复合控制器的逻辑：

```
self.joint_action_policy = IKSolverMink(...)
```

步骤 4：

在 `teleop_mink.py` 中，我们导入：

```
from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK
```

步骤 5：

在 `robosuite/examples/third_party_controller/default_mink_ik_gr1.json` 中，我们添加特定于新复合控制器的配置，并将 `type` 设置为
与 `WholeBodyMinkIK` 中指定的 `name` 匹配：

```
{
    "type": "WHOLE_BODY_MINK_IK",  # 设置正确的类型
    "composite_controller_specific_configs": {
            ...
    },
    ...
}
```
