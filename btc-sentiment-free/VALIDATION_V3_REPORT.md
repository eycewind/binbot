# Frozen v3 fresh-sample validation report

The selection and reference labels were committed in `d2157aba` before this inference run. The references are agent-reviewed labels, not independently established human ground truth. No prompt or label was changed after viewing these outputs.

## Run and denominators

- Run ID: `3013ca1fb65ce1a0c7ec8fef7ebc813cbe8640fe421eecb3a9737f22c20dc8dc`
- Model/digest: `qwen3.5:9b` / `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`
- Structural validity: **48/48**. Failed responses: **0**; all 48 selected records remain in structural, inclusion, scope, and horizon denominators.
- Inclusion/exclusion agreement: **45/48**.
- Sentiment-label agreement on reference-included articles: **39/43**. A failed or excluded model response to a reference-included article would count as disagreement.
- Exact scope-subtype agreement: **42/48**.
- Horizon agreement: **36/48**.
- Ambiguous reference items: **6**. Reference-excluded items: **5**; model-excluded items: **2**.

Agreement for an ambiguous item means agreement with at least one pre-registered accepted outcome. Scope-subtype agreement is stricter than inclusion agreement: `usd_explicit` and `btc_general` both enter the daily calculation. Horizon does not enter it. Positive/negative contribute ±1 and enter the denominator; neutral/mixed contribute 0 and enter it; uncertain or excluded items do not enter it.

## Targeted regressions

The ten expectations in `benchmark_v3_targeted.json` were committed before inference. Frozen v3 returned 10/10 structurally valid responses and matched an accepted pre-registered outcome on **8/10**. The deliberately ambiguous timing case matched one of its two accepted outcomes.

The two failures reproduce known scope patterns:

- `target_02_analogy`: expected `unrelated/uncertain/current_or_past`, got `relative_only/uncertain/current_or_past`. “Silver to bitcoin's gold” is an analogy, not actual BTC relative performance. Both outcomes are excluded, so this does not change the daily calculation.
- `target_03_steady_usd`: expected `usd_explicit/neutral/current_or_past`, got `btc_general/neutral/current_or_past`. “Unchanged in USD” and “dollar terms” explicitly denominate BTC price. Both outcomes are included and neutral, so this does not change the daily calculation.

## Every substantive disagreement

There are **19** items with at least one assessed dimension disagreement. Full supplied text is reproduced below; no article-body or external information was used.

### 2. Microstrategy Gears Up for Next Bitcoin Expansion With Fresh Capital Lined Up

- Disagreed dimensions: horizon
- Reference: `btc_general/positive/both`
- Frozen v3: `btc_general/positive/forward`
- Headline: Microstrategy Gears Up for Next Bitcoin Expansion With Fresh Capital Lined Up
- Summary: Microstrategy unveiled plans to issue preferred stock to fuel bitcoin acquisitions, following its $1.1 billion BTC purchase, solidifying its aggressive digital asset investment strategy.
- Reference reasoning: A completed BTC purchase and planned acquisitions directly increase BTC demand.
- v3 reasoning: Text describes MicroStrategy acquiring more Bitcoin with fresh capital, indicating increased demand and an aggressive investment strategy without specifying a USD price level.

### 6. Texas Senate Approves Landmark Bitcoin Reserve Bill—A Game-Changer for Digital Assets

- Disagreed dimensions: horizon
- Reference: `btc_general/positive/current_or_past`
- Frozen v3: `btc_general/positive/forward`
- Headline: Texas Senate Approves Landmark Bitcoin Reserve Bill—A Game-Changer for Digital Assets
- Summary: The Texas Senate has passed a bill to establish a state-run bitcoin reserve—cementing the state's crypto dominance and setting a precedent nationwide.
- Reference reasoning: The passed state reserve bill expands institutional adoption.
- v3 reasoning: The bill establishes a Bitcoin reserve, signaling increased adoption and demand. This is explicitly positive for BTC without providing an explicit USD price level.

### 8. Bitcoin Price Watch: Key Resistance at $95.5K Holds the Next Bullish Trigger

- Disagreed dimensions: horizon
- Reference: `usd_explicit/positive/both`
- Frozen v3: `usd_explicit/positive/forward`
- Headline: Bitcoin Price Watch: Key Resistance at $95.5K Holds the Next Bullish Trigger
- Summary: Bitcoin is consolidating just beneath its peak, following a sharp rally that pushed it to the upper bounds of this week’s trading range.
- Reference reasoning: The text gives a dollar resistance level, a completed rally, and a forward bullish trigger.
- v3 reasoning: Headline identifies $95.5K as a 'Bullish Trigger' for Bitcoin, implying an expected price rise above this level.

### 9. Metaplanet Adds 555 BTC to Its Treasury

- Disagreed dimensions: horizon
- Reference: `btc_general/positive/current_or_past`
- Frozen v3: `btc_general/positive/forward`
- Headline: Metaplanet Adds 555 BTC to Its Treasury
- Summary: The Japanese firm is now already more than halfway toward meeting its 10,000 BTC goal by the end of December 2025.
- Reference reasoning: The firm completed a 555 BTC treasury purchase.
- v3 reasoning: Metaplanet's acquisition adds significant demand, and the summary confirms progress toward a 10k BTC treasury goal by Dec 2025.

### 13. Cryptoquant Reports Surge in Bitcoin, ETH Selling Pressure as Prices Set Records

- Disagreed dimensions: sentiment label, horizon
- Reference: `usd_explicit/mixed/current_or_past`
- Frozen v3: `usd_explicit/negative/forward`
- Headline: Cryptoquant Reports Surge in Bitcoin, ETH Selling Pressure as Prices Set Records
- Summary: Large bitcoin holders and miners increased transfers to exchanges following bitcoin's climb to a new all-time high near $123K last week.
- Reference reasoning: BTC reached a dollar-denominated record while identified holder/miner transfers indicate selling pressure.
- v3 reasoning: Headline mentions selling pressure; summary notes large holders/miners transferring to exchanges after a price climb, indicating distribution/bearish sentiment despite the recent rise.

### 15. Bitcoin Is Down and Treasury Firms Are Buying the Dip

- Disagreed dimensions: horizon
- Reference: `btc_general/mixed/current_or_past`
- Frozen v3: `btc_general/mixed/both`
- Headline: Bitcoin Is Down and Treasury Firms Are Buying the Dip
- Summary: Institutions have remained surprisingly bullish on bitcoin, using their deep pockets to capitalize on buying opportunities during price dips.
- Reference reasoning: BTC is down while treasury firms are buying it, preserving negative price and positive demand signals.
- v3 reasoning: Headline indicates a bearish price move ('Down'), while summary states institutions are bullish on demand/adoption during the dip. Both signals coexist.

### 17. BTC Derivatives Week: Futures Heavy, Calls in Charge as $115K Acts Like Home

- Disagreed dimensions: scope subtype
- Reference: `usd_explicit/positive/current_or_past`; `usd_explicit/neutral/current_or_past`
- Frozen v3: `btc_general/neutral/current_or_past`
- Headline: BTC Derivatives Week: Futures Heavy, Calls in Charge as $115K Acts Like Home
- Summary: Explore the week's bitcoin derivatives markets with high positioning in futures and options, revealing important trading dynamics.
- Reference reasoning: The dollar price level is explicit. 'Calls in charge' can be read as bullish options positioning, while the summary otherwise states only trading dynamics.
- v3 reasoning: Headline mentions $115K price level but describes market positioning ('Calls in Charge') rather than a directional outlook. Summary is generic.

### 18. Blackrock’s IBIT Overtakes Deribit as World’s Largest Bitcoin Options Venue

- Disagreed dimensions: sentiment label, horizon
- Reference: `btc_general/neutral/current_or_past`
- Frozen v3: `btc_general/positive/forward`
- Headline: Blackrock’s IBIT Overtakes Deribit as World’s Largest Bitcoin Options Venue
- Summary: Blackrock’s Ishares Bitcoin Trust (IBIT) has surpassed Deribit in bitcoin options open interest, signaling a structural shift in BTC trading.
- Reference reasoning: A completed change in BTC options venue leadership is relevant but has no direct directional demand or price claim.
- v3 reasoning: Surpassing as the world's largest options venue signals increased institutional demand and structural growth for BTC, implying a positive outlook without specifying an explicit USD price level.

### 21. Bitcoin Is Floundering, but Saylor Is Digging in His Heels

- Disagreed dimensions: horizon
- Reference: `btc_general/negative/current_or_past`; `btc_general/mixed/both`
- Frozen v3: `btc_general/mixed/forward`
- Headline: Bitcoin Is Floundering, but Saylor Is Digging in His Heels
- Summary: "You need a 4-year time horizon and you need to be prepared to handle the volatility in this market,” said Strategy Chairman Michael Saylor.
- Reference reasoning: 'Floundering' is current negative BTC direction. Saylor's persistence and four-year advice may add a forward positive signal, but no explicit rise or purchase is stated.
- v3 reasoning: Headline implies bearish sentiment ('Floundering'), while summary cites a bullish long-term outlook from Saylor. Both signals are present.

### 25. Bitcoin (BTC) Price Hits $105K, Snaps Downtrend as Anticipation Soars for Donald Trump's Inauguration

- Disagreed dimensions: horizon
- Reference: `usd_explicit/positive/current_or_past`
- Frozen v3: `usd_explicit/positive/forward`
- Headline: Bitcoin (BTC) Price Hits $105K, Snaps Downtrend as Anticipation Soars for Donald Trump's Inauguration
- Summary: Crypto investors have high hopes for Donald Trump's presidency, including potential digital asset-focused executive orders that could add fuel to the rally.
- Reference reasoning: BTC hit $105K and snapped a downtrend.
- v3 reasoning: Headline cites BTC price hitting $105K and summary expresses high hopes for a rally driven by anticipated Trump policies.

### 27. Crypto Daybook Americas: Inflation Data May Shake Out Bitcoin's Doldrums as Demand for BTC Picks Up

- Disagreed dimensions: horizon
- Reference: `btc_general/positive/both`
- Frozen v3: `btc_general/positive/forward`
- Headline: Crypto Daybook Americas: Inflation Data May Shake Out Bitcoin's Doldrums as Demand for BTC Picks Up
- Summary: The day ahead in crypto: Feb. 12, 2025
- Reference reasoning: BTC demand is currently picking up and future inflation data may move it out of doldrums.
- v3 reasoning: Headline explicitly states 'Demand for BTC Picks Up', indicating increased demand. While inflation data is mentioned, the specific signal regarding Bitcoin's outlook is positive.

### 29. Solana News: Solana CME Futures Fall Short of Bitcoin (BTC) and Ethereum's ETH Debut, But There's a Catch

- Disagreed dimensions: inclusion/exclusion, scope subtype
- Reference: `relative_only/uncertain/current_or_past`
- Frozen v3: `btc_general/neutral/current_or_past`
- Headline: Solana News: Solana CME Futures Fall Short of Bitcoin (BTC) and Ethereum's ETH Debut, But There's a Catch
- Summary: Institutional demand on the first trading day fell way short of bitcoin's and ether's CME debut, but it was roughly in line when adjusted for the assets' market capitalization, K33 Research noted.
- Reference reasoning: The only BTC claim is Solana demand relative to BTC's earlier futures debut.
- v3 reasoning: Text compares Solana's institutional demand to Bitcoin and Ethereum but notes no specific price direction or USD-denominated outlook for BTC; it only mentions market cap adjustment.

### 30. China, Germany Fire Fiscal Rockets as U.S. Looks to Cut Spending. What Does it Mean for Bitcoin?

- Disagreed dimensions: inclusion/exclusion, scope subtype
- Reference: `uncertain/uncertain/forward`
- Frozen v3: `btc_general/positive/forward`
- Headline: China, Germany Fire Fiscal Rockets as U.S. Looks to Cut Spending. What Does it Mean for Bitcoin?
- Summary: China and Germany's shift in fiscal policy might calm crypto market nerves.
- Reference reasoning: The question and generic calming of crypto nerves do not establish a decisive BTC claim.
- v3 reasoning: Summary suggests fiscal policy shifts will calm market nerves, implying a positive outlook for Bitcoin demand/stability without explicit USD price quotes.

### 35. Bitcoin, XRP News: BTC Tops $108K on JPMorgan Crypto Filing, XRP Rallies on ETF News

- Disagreed dimensions: sentiment label, scope subtype
- Reference: `usd_explicit/positive/current_or_past`
- Frozen v3: `btc_general/neutral/current_or_past`
- Headline: Bitcoin, XRP News: BTC Tops $108K on JPMorgan Crypto Filing, XRP Rallies on ETF News
- Summary: The crypto market was unfazed by the escalating conflict in the Middle East on Monday.
- Reference reasoning: BTC topped an explicit $108K level.
- v3 reasoning: Headline mentions BTC price level ($108K) but summary only states market was 'unfazed' by conflict, offering no directional signal for BTC USD value or demand.

### 36. BTC Bounces After War-Driven Dip, Eyes $98.2K as Key Bull Market Line

- Disagreed dimensions: horizon
- Reference: `usd_explicit/positive/current_or_past`
- Frozen v3: `usd_explicit/positive/forward`
- Headline: BTC Bounces After War-Driven Dip, Eyes $98.2K as Key Bull Market Line
- Summary: Geopolitical tensions spark weekend volatility but BTC bounces back maintaining critical on-chain support.
- Reference reasoning: BTC bounced after a dip and maintained support; $98.2K supplies explicit USD scope.
- v3 reasoning: Headline explicitly mentions a price bounce and eyes $98.2K as a key bull market line, indicating an upward outlook.

### 38. Crypto Exchange Coinone Wins Court Battle Over Doubled Bitcoin Withdrawals in 2018 Network Chaos

- Disagreed dimensions: inclusion/exclusion, scope subtype
- Reference: `unrelated/uncertain/current_or_past`
- Frozen v3: `btc_general/neutral/current_or_past`
- Headline: Crypto Exchange Coinone Wins Court Battle Over Doubled Bitcoin Withdrawals in 2018 Network Chaos
- Summary: The court ruled the customers benefited from unjust enrichment due to a network delay, not the exchange’s servers.
- Reference reasoning: The claim concerns an exchange court dispute and customer enrichment, not BTC direction, demand, or adoption.
- v3 reasoning: Text describes a legal ruling regarding customer benefits from network delays, not BTC price direction or demand. No USD price level is mentioned.

### 41. Echoes of Summer 2023: Bitcoin’s Volatility Set to Surge

- Disagreed dimensions: sentiment label
- Reference: `btc_general/neutral/forward`
- Frozen v3: `btc_general/positive/forward`
- Headline: Echoes of Summer 2023: Bitcoin’s Volatility Set to Surge
- Summary: Market history suggests that prolonged IV compression often precedes explosive moves, with October once again shaping up as a likely turning point.
- Reference reasoning: A forecast volatility increase has no positive or negative direction.
- v3 reasoning: Headline notes volatility surge; summary implies explosive price moves ahead, indicating bullish outlook without specific USD levels.

### 42. Solana News: Bitcoin (BTC) Again Runs Into 2017-21 Trendline, SOL Flashes 'Shooting Star' Warning

- Disagreed dimensions: horizon
- Reference: `btc_general/negative/current_or_past`; `btc_general/neutral/current_or_past`
- Frozen v3: `btc_general/negative/forward`
- Headline: Solana News: Bitcoin (BTC) Again Runs Into 2017-21 Trendline, SOL Flashes 'Shooting Star' Warning
- Summary: Latest price moves from crypto’s big players show the bulls are hesitating ahead of the pivotal Fed rate decision.
- Reference reasoning: BTC running into a trendline and bulls hesitating can be bearish resistance, but the explicit shooting-star warning is attributed to SOL.
- v3 reasoning: Headline implies BTC is hitting a resistance trendline (bearish technical signal). Summary states bulls are 'hesitating' ahead of the Fed decision, indicating lack of bullish conviction or potential downside pressure.

### 43. Bitcoin Loses $106K as Bullish Crypto Bets Rack up $800M in Liquidations

- Disagreed dimensions: scope subtype
- Reference: `usd_explicit/negative/current_or_past`
- Frozen v3: `btc_general/negative/current_or_past`
- Headline: Bitcoin Loses $106K as Bullish Crypto Bets Rack up $800M in Liquidations
- Summary: Bitcoin accounted for roughly $344 million in losses, followed by Ether at $201 million, and Solana (SOL) at $97 million.
- Reference reasoning: BTC lost the explicit $106K level.
- v3 reasoning: Headline states Bitcoin lost value ($106K) and summary confirms losses, indicating a negative price move without explicit USD level context.
