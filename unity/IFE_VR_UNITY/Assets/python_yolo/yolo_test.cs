using UnityEditor;
using UnityEditor.Scripting.Python;

public class MenuItem_python_test_Class
{
   [MenuItem("Python Scripts/python_test")]
   public static void python_test()
   {
       PythonRunner.RunFile("Assets/python_yolo/yolo_test.py");
       }
};
