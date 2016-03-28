﻿using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.Research.MultiWorldTesting.ExploreLibrary
{
    public static class Decision
    {
        public static Decision<TAction, TExplorerState, TPolicyState> Create<TAction, TExplorerState, TPolicyState>(TAction action, TExplorerState explorerState, PolicyDecision<TAction, TPolicyState> policyDecision, bool shouldRecord)
        {
            return new Decision<TAction, TExplorerState, TPolicyState>
            {
                Action = action,
                ExplorerState = explorerState,
                PolicyDecision = policyDecision,
                ShouldRecord = shouldRecord
            };
        }
    }

    [JsonConverter(typeof(DecisionJsonConverter))]
    public sealed class Decision<TAction, TExplorerState, TPolicyState>
    {
        public bool ShouldRecord { get; set; }

        // int, int[]
        // choose action (shown)
        [JsonProperty("a")]
        public TAction Action { get; set; }

        // probability | predicted ranking, epsilon
        // "EpsilonGreedyLog":{ ... } 
        public TExplorerState ExplorerState { get; set; }

        // only logging TPolicyState
        public PolicyDecision<TAction, TPolicyState> PolicyDecision { get; set; }
    }
}
