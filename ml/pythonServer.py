import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linprog, minimize
from typing import List, Dict, Tuple, Union, Optional
import warnings
import logging
from datetime import datetime
import json
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import aiohttp
import asyncio
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import cvxpy as cp
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Initialize FastAPI app
app = FastAPI(title="Financial Management API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_management_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model 1: FinancialMDP
class FinancialMDP:
    def __init__(self, states: List[str], actions: List[str], transition_matrix: Dict, rewards: Dict[str, float], gamma: float = 0.95):
        self.states = states
        self.actions = actions
        self.transition_matrix = transition_matrix
        self.rewards = rewards
        self.gamma = gamma
        self.policy = {}

    def value_iteration(self, theta: float = 1e-6, max_iterations: int = 1000) -> Tuple[dict, dict]:
        V = {state: 0 for state in self.states}
        iteration = 0
        while iteration < max_iterations:
            delta = 0
            for state in self.states:
                v = V[state]
                action_values = []
                for action in self.actions:
                    value = 0
                    for next_state in self.states:
                        prob = self.transition_matrix[state][action][next_state]
                        value += prob * (self.rewards[state] + self.gamma * V[next_state])
                    action_values.append(value)
                V[state] = max(action_values)
                self.policy[state] = self.actions[np.argmax(action_values)]
                delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break
            iteration += 1
        return V, self.policy

# Model 2: RetirementOptimizer
class RetirementOptimizer:
    def __init__(self, assets: List[str], expected_returns: List[float], risk_levels: List[float], correlation_matrix: np.ndarray):
        self.assets = assets
        self.returns = np.array(expected_returns)
        self.risks = np.array(risk_levels)
        self.correlation = correlation_matrix

    def optimize_portfolio(self, target_return: float, risk_tolerance: float) -> Dict:
        n_assets = len(self.assets)
        def objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(np.diag(self.risks) @ self.correlation @ np.diag(self.risks), weights)))
            return portfolio_risk
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x: np.sum(x * self.returns) - target_return},
        ]
        bounds = [(0, 1) for _ in range(n_assets)]
        initial_weights = np.array([1 / n_assets] * n_assets)
        result = minimize(objective, initial_weights, method="SLSQP", constraints=constraints, bounds=bounds)
        return {
            "allocations": dict(zip(self.assets, result.x)),
            "expected_return": target_return,
            "portfolio_risk": result.fun,
            "sharpe_ratio": (target_return - 0.02) / result.fun,
        }

# Model 3: DebtManager
class DebtManager:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self._initialize_with_sample_data()

    def _initialize_with_sample_data(self) -> None:
        np.random.seed(42)
        n_samples = 1000
        income = np.random.normal(70000, 20000, n_samples)
        debt = np.random.normal(30000, 15000, n_samples)
        monthly_expenses = np.random.normal(3000, 1000, n_samples)
        credit_score = np.random.normal(700, 50, n_samples)
        data = pd.DataFrame({
            "income": income,
            "debt": debt,
            "debt_to_income_ratio": debt / income,
            "monthly_expenses": monthly_expenses,
            "credit_score": credit_score,
        })
        prob_success = 1 / (1 + np.exp(-(
            (data["income"] - np.mean(income)) / np.std(income) * 0.5
            + -(data["debt"] - np.mean(debt)) / np.std(debt) * 0.3
            + -(data["debt_to_income_ratio"] - np.mean(data["debt_to_income_ratio"])) / np.std(data["debt_to_income_ratio"]) * 0.4
            + -(data["monthly_expenses"] - np.mean(monthly_expenses)) / np.std(monthly_expenses) * 0.2
            + (data["credit_score"] - np.mean(credit_score)) / np.std(credit_score) * 0.3
        )))
        data["repayment_success"] = (np.random.random(n_samples) < prob_success).astype(int)
        self.train_model(data)

    def train_model(self, data: pd.DataFrame) -> None:
        features = ["income", "debt", "debt_to_income_ratio", "monthly_expenses", "credit_score"]
        X = data[features]
        y = data["repayment_success"]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_repayment_probability(self, financial_data: Dict) -> float:
        features = np.array([[
            financial_data["income"],
            financial_data["debt"],
            financial_data["debt"] / financial_data["income"],
            financial_data["monthly_expenses"],
            financial_data["credit_score"],
        ]])
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[0][1]

    def recommend_debt_strategy(self, financial_data: Dict) -> Dict:
        repayment_prob = self.predict_repayment_probability(financial_data)
        monthly_disposable = financial_data["income"] - financial_data["monthly_expenses"]
        if repayment_prob >= 0.7:
            strategy = "aggressive_repayment"
            monthly_payment = monthly_disposable * 0.5
        elif repayment_prob >= 0.4:
            strategy = "balanced_repayment"
            monthly_payment = monthly_disposable * 0.3
        else:
            strategy = "conservative_repayment"
            monthly_payment = monthly_disposable * 0.2
        return {
            "recommended_strategy": strategy,
            "suggested_monthly_payment": monthly_payment,
            "repayment_probability": repayment_prob,
            "estimated_payoff_months": financial_data["debt"] / monthly_payment,
        }

# Model 4: InvestmentLearningSystem
@dataclass
class UserProfile:
    age: int
    income: float
    risk_tolerance: float
    investment_horizon: int
    financial_knowledge: int
    investment_goals: List[str]

class InvestmentLearningSystem:
    def __init__(self, config_path: Optional[str] = None):
        self.setup_logging()
        self.load_config(config_path)
        self.initialize_models()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('investment_learning.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: Optional[str] = None):
        default_config = {
            'clustering': {
                'n_clusters': 5,
                'linkage': 'ward'
            },
            'simulation': {
                'default_days': 252,
                'default_simulations': 10000,
                'confidence_intervals': [0.95, 0.99]
            },
            'learning_paths': {
                'beginner': ['basics', 'risk', 'diversification'],
                'intermediate': ['asset_classes', 'portfolio_theory', 'market_analysis'],
                'advanced': ['derivatives', 'alternative_investments', 'trading_strategies']
            }
        }
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}. Using defaults.")
                self.config = default_config
        else:
            self.config = default_config

    def initialize_models(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.clustering = AgglomerativeClustering(
            n_clusters=self.config['clustering']['n_clusters'],
            linkage=self.config['clustering']['linkage']
        )

    def process_user_data(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = ['Age', 'Income', 'RiskTolerance']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        data = data.copy()
        data = data.fillna(data.mean())
        data['IncomeCategory'] = pd.qcut(data['Income'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['20s', '30s', '40s', '50s', '60+'])
        data['RiskCategory'] = pd.qcut(data['RiskTolerance'], q=3, labels=['Conservative', 'Moderate', 'Aggressive'])
        return data

    def cluster_users(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        features = ['Age', 'Income', 'RiskTolerance']
        X = self.scaler.fit_transform(data[features])
        X_pca = self.pca.fit_transform(X)
        clusters = self.clustering.fit_predict(X)
        data['Cluster'] = clusters
        data['PCA1'] = X_pca[:, 0]
        data['PCA2'] = X_pca[:, 1]
        metrics = self.calculate_cluster_metrics(data)
        return data, metrics

    def calculate_cluster_metrics(self, data: pd.DataFrame) -> Dict:
        metrics = {}
        for cluster in data['Cluster'].unique():
            cluster_data = data[data['Cluster'] == cluster]
            metrics[f'cluster_{cluster}'] = {
                'size': len(cluster_data),
                'avg_age': float(cluster_data['Age'].mean()),
                'avg_income': float(cluster_data['Income'].mean()),
                'avg_risk': float(cluster_data['RiskTolerance'].mean()),
                'risk_distribution': cluster_data['RiskCategory'].value_counts().to_dict()
            }
        return metrics

    def run_monte_carlo(self, profile: UserProfile) -> Dict:
        risk_factor = profile.risk_tolerance
        mu = 0.07 + (risk_factor - 0.5) * 0.05
        sigma = 0.15 + (risk_factor - 0.5) * 0.1
        results = self.monte_carlo_simulation(
            initial_investment=profile.income * 0.2,
            days=252 * profile.investment_horizon,
            mu=mu,
            sigma=sigma
        )
        metrics = self.calculate_simulation_metrics(results)
        return {
            'results': results.tolist(),
            'metrics': metrics,
            'parameters': {
                'mu': mu,
                'sigma': sigma,
                'horizon': profile.investment_horizon
            }
        }

    def monte_carlo_simulation(self, initial_investment: float, days: int, mu: float, sigma: float) -> np.ndarray:
        n_sims = self.config['simulation']['default_simulations']
        results = np.zeros((n_sims, days))
        for i in range(n_sims):
            random_normal = np.random.standard_t(df=5, size=days)
            volatility = np.exp(np.random.normal(0, 0.1, days))
            daily_returns = mu/days + sigma/np.sqrt(days) * random_normal * volatility
            cumulative_returns = np.cumprod(1 + daily_returns)
            results[i] = initial_investment * cumulative_returns
        return results[:, -1]

    def calculate_simulation_metrics(self, results: np.ndarray) -> Dict:
        metrics = {
            'mean': float(np.mean(results)),
            'std': float(np.std(results)),
            'median': float(np.median(results)),
            'var_95': float(np.percentile(results, 5)),
            'var_99': float(np.percentile(results, 1)),
            'max_drawdown': float(np.min(results)/np.max(results) - 1),
            'skewness': float(pd.Series(results).skew()),
            'kurtosis': float(pd.Series(results).kurt())
        }
        return metrics

    def generate_learning_path(self, profile: UserProfile) -> Dict:
        if profile.financial_knowledge <= 2:
            level = 'beginner'
        elif profile.financial_knowledge <= 4:
            level = 'intermediate'
        else:
            level = 'advanced'
        path = self.config['learning_paths'][level]
        if 'retirement' in profile.investment_goals:
            path.append('retirement_planning')
        if profile.risk_tolerance > 0.7:
            path.append('advanced_risk_management')
        return {
            'level': level,
            'path': path,
            'estimated_duration': len(path) * 2,
            'prerequisites': self.get_prerequisites(path)
        }

    def get_prerequisites(self, topics: List[str]) -> Dict[str, List[str]]:
        prereqs = {
            'portfolio_theory': ['basics', 'risk'],
            'derivatives': ['portfolio_theory', 'market_analysis'],
            'trading_strategies': ['market_analysis', 'risk']
        }
        return {topic: prereqs.get(topic, []) for topic in topics}

    def visualize_results(self, data: pd.DataFrame, simulation_results: Dict) -> None:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('User Clusters', 'Risk Distribution', 'Monte Carlo Simulation', 'Learning Progress')
        )
        fig.add_trace(
            go.Scatter(
                x=data['PCA1'],
                y=data['PCA2'],
                mode='markers',
                marker=dict(
                    color=data['Cluster'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=data['RiskCategory'],
                name='Clusters'
            ),
            row=1, col=1
        )
        risk_dist = data['RiskCategory'].value_counts()
        fig.add_trace(
            go.Bar(
                x=risk_dist.index,
                y=risk_dist.values,
                name='Risk Distribution'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(
                x=simulation_results['results'],
                name='Portfolio Returns'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                y=[1, 2, 4, 7],
                mode='lines+markers',
                name='Learning Progress'
            ),
            row=2, col=2
        )
        fig.update_layout(height=800, showlegend=True)
        fig.show()

# Model 5: MarketSentimentAnalyzer
@dataclass
class NewsSource:
    name: str
    api_endpoint: str
    api_key: str
    weight: float = 1.0

class MarketSentimentAnalyzer:
    def __init__(self, config_path: Optional[str] = None):
        self.setup_logging()
        self.load_config(config_path)
        self.initialize_models()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sentiment_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: Optional[str] = None):
        default_config = {
            'news_sources': [
                {
                    'name': 'Alpha Vantage News',
                    'api_endpoint': 'https://www.alphavantage.co/query',
                    'api_key': 'NA5WJYEDMIZN4W87',
                    'weight': 1.0
                },
                {
                    'name': 'NewsAPI',
                    'api_endpoint': 'https://newsapi.org/v2/everything',
                    'api_key': '92f6ae92547a4d7a82200c6f22b9db95',
                    'weight': 1.0
                }
            ],
            'sentiment_thresholds': {
                'very_negative': 0.2,
                'negative': 0.4,
                'neutral': 0.6,
                'positive': 0.8,
                'very_positive': 1.0
            },
            'update_interval': 3600,
            'max_articles_per_source': 50
        }
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}. Using defaults.")
                self.config = default_config
        else:
            self.config = default_config

    def initialize_models(self):
        model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.general_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.logger.info("Successfully initialized all NLP models")

    async def fetch_alpha_vantage_news(self, source: NewsSource) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            try:
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'apikey': source.api_key,
                    'topics': 'financial_markets'
                }
                async with session.get(source.api_endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('feed', [])
                        return [
                            {
                                'title': article.get('title', ''),
                                'text': article.get('summary', ''),
                                'source': 'Alpha Vantage',
                                'url': article.get('url', ''),
                                'timestamp': article.get('time_published', '')
                            }
                            for article in articles[:self.config['max_articles_per_source']]
                        ]
                    else:
                        self.logger.warning(f"Failed to fetch news from Alpha Vantage: {response.status}")
                        return []
            except Exception as e:
                self.logger.error(f"Error fetching from Alpha Vantage: {e}")
                return []

    async def fetch_newsapi_news(self, source: NewsSource) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            try:
                params = {
                    'q': 'finance OR stock market OR economy',
                    'apiKey': source.api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt'
                }
                async with session.get(source.api_endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        return [
                            {
                                'title': article.get('title', ''),
                                'text': article.get('description', ''),
                                'source': 'NewsAPI',
                                'url': article.get('url', ''),
                                'timestamp': article.get('publishedAt', '')
                            }
                            for article in articles[:self.config['max_articles_per_source']]
                        ]
                    else:
                        self.logger.warning(f"Failed to fetch news from NewsAPI: {response.status}")
                        return []
            except Exception as e:
                self.logger.error(f"Error fetching from NewsAPI: {e}")
                return []

    async def fetch_news(self, source: NewsSource) -> List[Dict]:
        if source.name == 'Alpha Vantage News':
            return await self.fetch_alpha_vantage_news(source)
        elif source.name == 'NewsAPI':
            return await self.fetch_newsapi_news(source)
        return []

    def analyze_sentiment(self, text: str) -> Dict:
        if not text:
            return self._get_default_sentiment()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        fin_sentiment = outputs.logits.softmax(dim=1).detach().numpy()[0]
        gen_sentiment = self.general_sentiment(text)[0]
        return {
            'financial_sentiment': {
                'positive': float(fin_sentiment[0]),
                'negative': float(fin_sentiment[1]),
                'neutral': float(fin_sentiment[2])
            },
            'general_sentiment': {
                'label': gen_sentiment['label'],
                'score': float(gen_sentiment['score'])
            }
        }

    def _get_default_sentiment(self) -> Dict:
        return {
            'financial_sentiment': {
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34
            },
            'general_sentiment': {
                'label': 'NEUTRAL',
                'score': 0.5
            }
        }

    async def analyze_market_sentiment(self) -> Dict:
        sources = [NewsSource(**s) for s in self.config['news_sources']]
        tasks = [self.fetch_news(source) for source in sources]
        all_articles = await asyncio.gather(*tasks)
        articles = [article for source_articles in all_articles for article in source_articles]
        if not articles:
            self.logger.warning("No articles found to analyze")
            return {'overall_sentiment': self._get_default_sentiment()}
        analyses = []
        for article in articles:
            text = f"{article['title']} {article['text']}"
            sentiment = self.analyze_sentiment(text)
            analyses.append({
                'article': article,
                'sentiment': sentiment
            })
        overall_sentiment = self.aggregate_sentiment(analyses)
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_sentiment': overall_sentiment,
            'articles_analyzed': len(analyses),
            'analyses': analyses
        }

    def aggregate_sentiment(self, analyses: List[Dict]) -> Dict:
        if not analyses:
            return self._get_default_sentiment()
        financial_sentiment = {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0
        }
        for analysis in analyses:
            sentiment = analysis['sentiment']['financial_sentiment']
            for k, v in sentiment.items():
                financial_sentiment[k] += v
        total = len(analyses)
        return {
            'financial_sentiment': {
                k: v/total for k, v in financial_sentiment.items()
            }
        }

# Model 6: PortfolioOptimizer
@dataclass
class PortfolioConstraints:
    min_weight: float = 0.0
    max_weight: float = 1.0
    sector_constraints: Dict[str, Tuple[float, float]] = None
    turnover_limit: float = None
    risk_budget: Dict[str, float] = None

class PortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('portfolio_optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, tickers: List[str], period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        try:
            data = yf.download(tickers, period=period, interval=interval)['Close']
            if data.empty:
                raise ValueError("No data retrieved for specified tickers")
            missing_pct = data.isnull().mean() * 100
            if missing_pct.max() > 10:
                self.logger.warning(f"High missing data percentage: {missing_pct}")
            data = data.fillna(method='ffill')
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            raise

    def calculate_metrics(self, returns: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        expected_returns = returns.ewm(span=252).mean().iloc[-1].to_numpy()
        cov_matrix = returns.ewm(span=252).cov().iloc[-252:]
        cov_matrix = cov_matrix.groupby(level=0).tail(1).to_numpy()
        min_eigenval = np.min(np.linalg.eigvals(cov_matrix))
        if min_eigenval < 0:
            cov_matrix -= 1.1 * min_eigenval * np.eye(cov_matrix.shape[0])
        return expected_returns, cov_matrix

    def calculate_risk_metrics(self, weights: np.ndarray, returns: pd.DataFrame, cov_matrix: np.ndarray) -> Dict:
        portfolio_returns = returns @ weights
        annual_return = np.mean(portfolio_returns) * 252
        annual_vol = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol
        var_95 = norm.ppf(0.05, annual_return, annual_vol)
        cvar_95 = -np.mean(portfolio_returns[portfolio_returns <= var_95])
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns/rolling_max - 1
        max_drawdown = drawdowns.min()
        return {
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            'sharpe_ratio': float(sharpe_ratio),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'max_drawdown': float(max_drawdown)
        }

    def optimize_portfolio(self, tickers: List[str], risk_aversion: float, constraints: PortfolioConstraints = None, current_weights: np.ndarray = None) -> Dict:
        try:
            prices = self.fetch_data(tickers)
            returns = prices.pct_change().dropna()
            expected_returns, cov_matrix = self.calculate_metrics(returns)
            n_assets = len(tickers)
            weights = cp.Variable(n_assets)
            returns_term = expected_returns @ weights
            risk_term = cp.quad_form(weights, cov_matrix)
            objective = cp.Maximize(returns_term - risk_aversion * risk_term)
            constraints = [
                cp.sum(weights) == 1,
                weights >= constraints.min_weight if constraints else 0,
                weights <= constraints.max_weight if constraints else 1
            ]
            if current_weights is not None and constraints.turnover_limit:
                turnover = cp.sum(cp.abs(weights - current_weights))
                constraints.append(turnover <= constraints.turnover_limit)
            problem = cp.Problem(objective, constraints)
            problem.solve()
            if problem.status != 'optimal':
                raise ValueError(f"Optimization failed with status: {problem.status}")
            optimal_weights = weights.value
            risk_metrics = self.calculate_risk_metrics(optimal_weights, returns, cov_matrix)
            results = {
                'weights': dict(zip(tickers, optimal_weights)),
                'risk_metrics': risk_metrics,
                'optimization_status': problem.status,
                'timestamp': datetime.now().isoformat()
            }
            self.logger.info("Portfolio optimization completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            raise

    def plot_portfolio_analysis(self, results: Dict, save_path: Optional[str] = None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        weights = pd.Series(results['weights'])
        weights.plot(kind='bar', ax=ax1)
        ax1.set_title('Portfolio Allocation')
        ax1.set_ylabel('Weight')
        ax1.tick_params(axis='x', rotation=45)
        risk_metrics = pd.Series(results['risk_metrics'])
        risk_metrics.plot(kind='bar', ax=ax2)
        ax2.set_title('Risk Metrics')
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

# Model 7: RiskAssessmentEngine
class RiskAssessmentEngine:
    def __init__(self, config_path: str = None):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.setup_logging()
        self.load_config(config_path)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('risk_assessment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str = None):
        default_config = {
            'risk_levels': {
                0: "Conservative",
                1: "Moderate",
                2: "Aggressive"
            },
            'feature_weights': {
                'risk_tolerance': 0.3,
                'investment_horizon': 0.3,
                'market_experience': 0.2,
                'income_stability': 0.1,
                'emergency_fund': 0.1
            },
            'market_condition_adjustments': {
                'bull_market': 0.1,
                'bear_market': -0.1,
                'neutral_market': 0
            }
        }
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config file: {e}. Using defaults.")
                self.config = default_config
        else:
            self.config = default_config

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        required_columns = [
            'risk_tolerance', 'investment_horizon', 'market_experience',
            'income_stability', 'emergency_fund'
        ]
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        data['risk_score'] = self.calculate_risk_score(data)
        data['market_adjusted_score'] = self.adjust_for_market_conditions(data['risk_score'])
        return data

    def calculate_risk_score(self, data: pd.DataFrame) -> pd.Series:
        risk_score = sum(
            data[feature] * weight
            for feature, weight in self.config['feature_weights'].items()
        )
        return risk_score

    def adjust_for_market_conditions(self, risk_scores: pd.Series) -> pd.Series:
        market_condition = self.get_market_condition()
        adjustment = self.config['market_condition_adjustments'][market_condition]
        return risk_scores * (1 + adjustment)

    def get_market_condition(self) -> str:
        return 'neutral_market'

    def fit(self, data: pd.DataFrame):
        try:
            processed_data = self.preprocess_data(data)
            scaled_data = self.scaler.fit_transform(processed_data.drop('risk_category', axis=1))
            self.kmeans.fit(scaled_data)
            processed_data['cluster'] = self.kmeans.labels_
            self.rf_classifier.fit(scaled_data, processed_data['cluster'])
            self.logger.info("Successfully fitted models on training data")
        except Exception as e:
            self.logger.error(f"Error during model fitting: {e}")
            raise

    def predict_risk(self, user_data: Union[List, Dict, pd.DataFrame]) -> Dict:
        try:
            if isinstance(user_data, list):
                user_data = pd.DataFrame([user_data], columns=list(self.config['feature_weights'].keys()))
            elif isinstance(user_data, dict):
                user_data = pd.DataFrame([user_data])
            processed_data = self.preprocess_data(user_data)
            scaled_data = self.scaler.transform(processed_data)
            kmeans_cluster = self.kmeans.predict(scaled_data)[0]
            rf_cluster = self.rf_classifier.predict(scaled_data)[0]
            rf_probabilities = self.rf_classifier.predict_proba(scaled_data)[0]
            risk_assessment = {
                'risk_category': self.config['risk_levels'][rf_cluster],
                'confidence_score': float(max(rf_probabilities)),
                'risk_score': float(processed_data['risk_score'].iloc[0]),
                'market_adjusted_score': float(processed_data['market_adjusted_score'].iloc[0]),
                'assessment_timestamp': datetime.now().isoformat(),
                'model_version': '1.0'
            }
            self.logger.info(f"Successfully generated risk assessment: {risk_assessment['risk_category']}")
            return risk_assessment
        except Exception as e:
            self.logger.error(f"Error during risk prediction: {e}")
            raise

    def save_models(self, path: str):
        try:
            joblib.dump({
                'scaler': self.scaler,
                'kmeans': self.kmeans,
                'rf_classifier': self.rf_classifier,
                'config': self.config
            }, path)
            self.logger.info(f"Successfully saved models to {path}")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            raise

    def load_models(self, path: str):
        try:
            models = joblib.load(path)
            self.scaler = models['scaler']
            self.kmeans = models['kmeans']
            self.rf_classifier = models['rf_classifier']
            self.config = models['config']
            self.logger.info(f"Successfully loaded models from {path}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

# Define Pydantic models for request/response validation
class UserRiskProfile(BaseModel):
    risk_tolerance: int = Field(..., ge=1, le=5, description="Risk tolerance on a scale of 1-5")
    investment_horizon: int = Field(..., ge=1, description="Investment horizon in years")
    market_experience: int = Field(..., ge=1, le=5, description="Market experience on a scale of 1-5")
    income_stability: int = Field(..., ge=1, le=5, description="Income stability on a scale of 1-5")
    emergency_fund: int = Field(..., ge=1, le=5, description="Emergency fund adequacy on a scale of 1-5")

class PortfolioOptimizationRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=2, description="List of stock tickers")
    risk_aversion: float = Field(..., ge=0.0, le=1.0, description="Risk aversion parameter")
    min_weight: float = Field(0.0, ge=0.0, le=1.0, description="Minimum weight per asset")
    max_weight: float = Field(1.0, ge=0.0, le=1.0, description="Maximum weight per asset")
    turnover_limit: Optional[float] = Field(None, ge=0.0, le=1.0, description="Maximum portfolio turnover")

class InvestmentLearningRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="User's age")
    income: float = Field(..., ge=0, description="Annual income")
    risk_tolerance: float = Field(..., ge=0.0, le=1.0, description="Risk tolerance")
    investment_horizon: int = Field(..., ge=1, description="Investment horizon in years")
    financial_knowledge: int = Field(..., ge=1, le=5, description="Financial knowledge on a scale of 1-5")
    investment_goals: List[str] = Field(..., min_items=1, description="List of investment goals")

class DebtManagementRequest(BaseModel):
    income: float = Field(..., ge=0, description="Annual income")
    debt: float = Field(..., ge=0, description="Total debt")
    monthly_expenses: float = Field(..., ge=0, description="Monthly expenses")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")

class LifeEventSimulationRequest(BaseModel):
    current_state: str = Field(..., description="Current life state")
    planning_horizon: int = Field(..., ge=1, le=30, description="Planning horizon in years")

class RetirementPlanningRequest(BaseModel):
    target_return: float = Field(..., ge=0.0, le=0.2, description="Target annual return")
    risk_tolerance: float = Field(..., ge=0.0, le=0.2, description="Risk tolerance")

# Initialize the models
financial_mdp = FinancialMDP(
    states=["Employed", "Unemployed", "Retired", "Self_Employed"],
    actions=["Save_Aggressive", "Save_Moderate", "Spend_Moderate", "Spend_Aggressive"],
    transition_matrix={
        "Employed": {
            "Save_Aggressive": {"Employed": 0.8, "Unemployed": 0.1, "Retired": 0.05, "Self_Employed": 0.05},
            "Save_Moderate": {"Employed": 0.75, "Unemployed": 0.15, "Retired": 0.05, "Self_Employed": 0.05},
            "Spend_Moderate": {"Employed": 0.7, "Unemployed": 0.2, "Retired": 0.05, "Self_Employed": 0.05},
            "Spend_Aggressive": {"Employed": 0.6, "Unemployed": 0.25, "Retired": 0.1, "Self_Employed": 0.05}
        },
        "Unemployed": {
            "Save_Aggressive": {"Employed": 0.3, "Unemployed": 0.5, "Retired": 0.1, "Self_Employed": 0.1},
            "Save_Moderate": {"Employed": 0.25, "Unemployed": 0.55, "Retired": 0.1, "Self_Employed": 0.1},
            "Spend_Moderate": {"Employed": 0.2, "Unemployed": 0.6, "Retired": 0.1, "Self_Employed": 0.1},
            "Spend_Aggressive": {"Employed": 0.15, "Unemployed": 0.65, "Retired": 0.1, "Self_Employed": 0.1}
        },
        "Retired": {
            "Save_Aggressive": {"Employed": 0.05, "Unemployed": 0.05, "Retired": 0.85, "Self_Employed": 0.05},
            "Save_Moderate": {"Employed": 0.05, "Unemployed": 0.05, "Retired": 0.85, "Self_Employed": 0.05},
            "Spend_Moderate": {"Employed": 0.05, "Unemployed": 0.1, "Retired": 0.8, "Self_Employed": 0.05},
            "Spend_Aggressive": {"Employed": 0.05, "Unemployed": 0.15, "Retired": 0.75, "Self_Employed": 0.05}
        },
        "Self_Employed": {
            "Save_Aggressive": {"Employed": 0.1, "Unemployed": 0.1, "Retired": 0.05, "Self_Employed": 0.75},
            "Save_Moderate": {"Employed": 0.15, "Unemployed": 0.1, "Retired": 0.05, "Self_Employed": 0.7},
            "Spend_Moderate": {"Employed": 0.2, "Unemployed": 0.15, "Retired": 0.05, "Self_Employed": 0.6},
            "Spend_Aggressive": {"Employed": 0.25, "Unemployed": 0.2, "Retired": 0.05, "Self_Employed": 0.5}
        }
    },
    rewards={"Employed": 100, "Unemployed": -50, "Retired": 30, "Self_Employed": 80}
)

retirement_optimizer = RetirementOptimizer(
    assets=["Stocks", "Bonds", "Real_Estate", "Cash"],
    expected_returns=[0.08, 0.04, 0.06, 0.02],
    risk_levels=[0.15, 0.05, 0.10, 0.01],
    correlation_matrix=np.array([
        [1.0, 0.2, 0.3, 0.0],
        [0.2, 1.0, 0.1, 0.0],
        [0.3, 0.1, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
)

debt_manager = DebtManager()

investment_system = InvestmentLearningSystem()

market_sentiment_analyzer = MarketSentimentAnalyzer()

portfolio_optimizer = PortfolioOptimizer()

risk_engine = RiskAssessmentEngine()

# Load sample data for risk assessment
sample_data = pd.DataFrame({
    'risk_tolerance': [1, 3, 5, 2, 4, 5, 3, 1, 2, 4],
    'investment_horizon': [5, 10, 15, 3, 8, 12, 6, 2, 4, 9],
    'market_experience': [1, 3, 4, 2, 5, 4, 3, 1, 2, 5],
    'income_stability': [3, 4, 5, 2, 4, 5, 3, 2, 3, 4],
    'emergency_fund': [2, 3, 4, 2, 3, 4, 3, 2, 2, 3],
    'risk_category': ['Conservative', 'Moderate', 'Aggressive', 'Conservative',
                     'Moderate', 'Aggressive', 'Moderate', 'Conservative',
                     'Conservative', 'Moderate']
})

# Train risk assessment model on sample data
risk_engine.fit(sample_data)

# API routes
@app.get("/")
async def root():
    return {
        "message": "Financial Management API",
        "available_endpoints": [
            "/risk-assessment",
            "/portfolio-optimization",
            "/market-sentiment",
            "/investment-learning",
            "/debt-management",
            "/life-event-simulation",
            "/retirement-planning"
        ]
    }

@app.post("/risk-assessment")
async def assess_risk(user_data: UserRiskProfile):
    try:
        user_dict = user_data.dict()
        assessment = risk_engine.predict_risk(user_dict)
        return assessment
    except Exception as e:
        logger.error(f"Error in risk assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio-optimization")
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    try:
        constraints = PortfolioConstraints(
            min_weight=request.min_weight,
            max_weight=request.max_weight,
            turnover_limit=request.turnover_limit
        )
        results = portfolio_optimizer.optimize_portfolio(
            tickers=request.tickers,
            risk_aversion=request.risk_aversion,
            constraints=constraints
        )
        return results
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-sentiment")
async def get_market_sentiment():
    try:
        sentiment_results = await market_sentiment_analyzer.analyze_market_sentiment()
        return sentiment_results
    except Exception as e:
        logger.error(f"Error in market sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/investment-learning")
async def get_investment_learning(request: InvestmentLearningRequest):
    try:
        user_profile = UserProfile(
            age=request.age,
            income=request.income,
            risk_tolerance=request.risk_tolerance,
            investment_horizon=request.investment_horizon,
            financial_knowledge=request.financial_knowledge,
            investment_goals=request.investment_goals
        )
        simulation_results = investment_system.run_monte_carlo(user_profile)
        learning_path = investment_system.generate_learning_path(user_profile)
        return {
            "simulation_results": simulation_results,
            "learning_path": learning_path
        }
    except Exception as e:
        logger.error(f"Error in investment learning system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debt-management")
async def manage_debt(request: DebtManagementRequest):
    try:
        financial_data = request.dict()
        strategy = debt_manager.recommend_debt_strategy(financial_data)
        return strategy
    except Exception as e:
        logger.error(f"Error in debt management: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/life-event-simulation")
async def simulate_life_events(request: LifeEventSimulationRequest):
    try:
        if request.current_state not in financial_mdp.states:
            raise HTTPException(status_code=400, detail=f"Invalid state. Must be one of {financial_mdp.states}")
        optimal_values, optimal_policy = financial_mdp.value_iteration()
        current_state = request.current_state
        future_states = []
        future_rewards = []
        for year in range(request.planning_horizon):
            action = optimal_policy[current_state]
            transition_probs = financial_mdp.transition_matrix[current_state][action]
            next_state = np.random.choice(
                financial_mdp.states,
                p=[transition_probs[s] for s in financial_mdp.states]
            )
            future_states.append({
                "year": year + 1,
                "state": next_state,
                "action": action,
                "reward": financial_mdp.rewards[next_state]
            })
            future_rewards.append(financial_mdp.rewards[next_state])
            current_state = next_state
        cumulative_rewards = np.cumsum(future_rewards)
        return {
            "optimal_policy": optimal_policy,
            "optimal_values": optimal_values,
            "future_states": future_states,
            "cumulative_rewards": cumulative_rewards.tolist(),
            "expected_lifetime_value": optimal_values[request.current_state]
        }
    except Exception as e:
        logger.error(f"Error in life event simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retirement-planning")
async def plan_retirement(request: RetirementPlanningRequest):
    try:
        portfolio = retirement_optimizer.optimize_portfolio(
            target_return=request.target_return,
            risk_tolerance=request.risk_tolerance
        )
        return portfolio
    except Exception as e:
        logger.error(f"Error in retirement planning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Main function to run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
