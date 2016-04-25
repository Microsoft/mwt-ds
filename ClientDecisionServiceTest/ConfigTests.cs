﻿using Microsoft.Research.MultiWorldTesting.ClientLibrary;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;
using System.Threading;

namespace ClientDecisionServiceTest
{
    [TestClass]
    public class ConfigTests : MockCommandTestBase
    {
        [TestMethod]
        public void TestSingleActionInvalidPathOutputDir()
        {
            joinServer.Reset();

            commandCenter.CreateBlobs(createSettingsBlob: true, createModelBlob: false);

            var dsConfig = new DecisionServiceConfiguration(MockCommandCenter.AuthorizationToken);
            dsConfig.JoinServerType = JoinServerType.CustomSolution;
            dsConfig.LoggingServiceAddress = MockJoinServer.MockJoinServerAddress;
            dsConfig.BlobOutputDir = @"c:\";
            dsConfig.PollingForSettingsPeriod = TimeSpan.FromMilliseconds(500);

            var cancelTokenSource = new CancellationTokenSource();
            bool exceptionIsExpected = false;

            dsConfig.SettingsPollFailureCallback = (ex) =>
            {
                if (ex is UnauthorizedAccessException)
                {
                    exceptionIsExpected = true;
                    cancelTokenSource.Cancel();
                }
            };

            using (var ds = DecisionService
                .WithPolicy(dsConfig)
                .With<TestContext>()
                .WithEpsilonGreedy(.2f)
                .ExploitUntilModelReady(new TestSingleActionPolicy()))
            {
                cancelTokenSource.Token.WaitHandle.WaitOne(5000);
            }

            Assert.AreEqual(true, exceptionIsExpected);
        }

        [TestMethod]
        public void TestSingleActionSettingsBlobOutput()
        {
            joinServer.Reset();

            string settingsPath = ".\\dstestsettings";
            Directory.CreateDirectory(settingsPath);

            commandCenter.CreateBlobs(createSettingsBlob: true, createModelBlob: false);

            var dsConfig = new DecisionServiceConfiguration(MockCommandCenter.AuthorizationToken);

            dsConfig.JoinServerType = JoinServerType.CustomSolution;
            dsConfig.LoggingServiceAddress = MockJoinServer.MockJoinServerAddress;
            dsConfig.BlobOutputDir = settingsPath;
            dsConfig.PollingForSettingsPeriod = TimeSpan.FromMilliseconds(500);

            using (var ds = DecisionService
                .WithPolicy(dsConfig)
                .With<TestContext>()
                .WithEpsilonGreedy(.2f)
                .ExploitUntilModelReady(new TestSingleActionPolicy()))
            {

                string settingsFile = Path.Combine(settingsPath, "settings-" + commandCenter.LocalAzureSettingsBlobName);

                int sleepCount = 20;
                while (true && sleepCount > 0)
                {
                    Thread.Sleep(100);
                    sleepCount--;

                    if (File.Exists(settingsFile))
                    {
                        break;
                    }
                }

                Assert.AreNotEqual(0, sleepCount);

                while (true)
                {
                    try
                    {
                        byte[] settingsBytes = File.ReadAllBytes(settingsFile);

                        Assert.IsTrue(Enumerable.SequenceEqual(settingsBytes, commandCenter.GetSettingsBlobContent()));
                        break;
                    }
                    catch (IOException) { }
                }
            }

            Directory.Delete(settingsPath, true);
        }

        [TestMethod]
        public void TestMultiActionInvalidPathOutputDir()
        {
            joinServer.Reset();

            commandCenter.CreateBlobs(createSettingsBlob: true, createModelBlob: false);

            var dsConfig = new DecisionServiceConfiguration(MockCommandCenter.AuthorizationToken);
            //explorer: new EpsilonGreedyExplorer<TestContext>(new TestMultiActionPolicy(), epsilon: 0.2f, numActions: 2));

            dsConfig.JoinServerType = JoinServerType.CustomSolution;
            dsConfig.LoggingServiceAddress = MockJoinServer.MockJoinServerAddress;
            dsConfig.BlobOutputDir = @"c:\";
            dsConfig.PollingForSettingsPeriod = TimeSpan.FromMilliseconds(500);

            var cancelTokenSource = new CancellationTokenSource();
            bool exceptionIsExpected = false;

            dsConfig.SettingsPollFailureCallback = (ex) =>
            {
                if (ex is UnauthorizedAccessException)
                {
                    exceptionIsExpected = true;
                    cancelTokenSource.Cancel();
                }
            };

            using (var ds = DecisionService
                .WithRanker(dsConfig)
                .With<TestContext>()
                .WithTopSlotEpsilonGreedy(.2f)
                .ExploitUntilModelReady(new TestMultiActionPolicy()))
            {
                cancelTokenSource.Token.WaitHandle.WaitOne(5000);
            }

            Assert.AreEqual(true, exceptionIsExpected);
        }

        [TestMethod]
        public void TestMultiActionSettingsBlobOutput()
        {
            joinServer.Reset();

            string settingsPath = ".\\dstestsettings";
            Directory.CreateDirectory(settingsPath);

            commandCenter.CreateBlobs(createSettingsBlob: true, createModelBlob: false);

            var dsConfig = new DecisionServiceConfiguration(MockCommandCenter.AuthorizationToken);
            dsConfig.JoinServerType = JoinServerType.CustomSolution;
            dsConfig.LoggingServiceAddress = MockJoinServer.MockJoinServerAddress;
            dsConfig.BlobOutputDir = settingsPath;
            dsConfig.PollingForSettingsPeriod = TimeSpan.FromMilliseconds(500);

            using (var ds = DecisionService
                .WithRanker(dsConfig)
                .With<TestContext>()
                .WithTopSlotEpsilonGreedy(.2f)
                .ExploitUntilModelReady(new TestMultiActionPolicy()))
            {
                string settingsFile = Path.Combine(settingsPath, "settings-" + commandCenter.LocalAzureSettingsBlobName);

                int sleepCount = 20;
                while (true && sleepCount > 0)
                {
                    Thread.Sleep(100);
                    sleepCount--;

                    if (File.Exists(settingsFile))
                    {
                        break;
                    }
                }

                Assert.AreNotEqual(0, sleepCount);

                while (true)
                {
                    try
                    {
                        byte[] settingsBytes = File.ReadAllBytes(settingsFile);

                        Assert.IsTrue(Enumerable.SequenceEqual(settingsBytes, commandCenter.GetSettingsBlobContent()));
                        break;
                    }
                    catch (IOException) { }
                }
            }
            Directory.Delete(settingsPath, true);
        }
    }
}