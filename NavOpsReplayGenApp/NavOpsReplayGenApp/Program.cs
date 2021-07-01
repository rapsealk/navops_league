using System;
using System.IO;

namespace NavOpsReplayGenApp
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = @"C:\Users\rapsealk\Desktop\NavOps";

            foreach (var arg in args)
            {
                Console.WriteLine($"Arg: {arg}");
                if (arg.StartsWith("--path"))
                {
                    string[] values = arg.Split('=');
                    path = values[values.Length - 1];
                }
            }
            Console.WriteLine($"Path: {path}");
            return;

            /*using*/
            var watcher = new FileSystemWatcher(path);

            watcher.NotifyFilter = NotifyFilters.Attributes
                                 | NotifyFilters.CreationTime
                                 | NotifyFilters.DirectoryName
                                 | NotifyFilters.FileName
                                 | NotifyFilters.LastAccess
                                 | NotifyFilters.LastWrite
                                 | NotifyFilters.Security
                                 | NotifyFilters.Size;

            watcher.Changed += OnChanged;
            watcher.Created += OnCreated;
            watcher.Deleted += OnDeleted;
            watcher.Renamed += OnRenamed;
            watcher.Error += OnError;

            watcher.Filter = "*.txt";
            watcher.IncludeSubdirectories = true;
            watcher.EnableRaisingEvents = true;

            Console.WriteLine("Press enter to exit.");
            Console.ReadLine();
        }

        private static void OnChanged(object sender, FileSystemEventArgs e)
        {
            if (e.ChangeType != WatcherChangeTypes.Changed)
            {
                return;
            }
            Console.WriteLine($"Changed: {e.FullPath}");
        }

        private static void OnCreated(object sender, FileSystemEventArgs e)
        {
            Console.WriteLine($"Created: {e.FullPath}");
        }

        private static void OnDeleted(object sender, FileSystemEventArgs e) =>
            Console.WriteLine($"Deleted: {e.FullPath}");

        private static void OnRenamed(object sender, RenamedEventArgs e)
        {
            Console.WriteLine("Renamed:");
            Console.WriteLine($"- Old: {e.OldFullPath}");
            Console.WriteLine($"- New: {e.FullPath}");
        }

        private static void OnError(object sender, ErrorEventArgs e) =>
            PrintException(e.GetException());

        private static void PrintException(Exception? e)
        {
            if (e != null)
            {
                Console.WriteLine($"Message: {e.Message}");
                Console.WriteLine("Stacktrace:");
                Console.WriteLine(e.StackTrace);
                Console.WriteLine();
                PrintException(e.InnerException);
            }
        }
    }
}
