"use client"
import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Target,
  Brain,
  PiggyBank,
  GraduationCap,
  ArrowRight,
  Shield,
  BarChart3,
  Sun,
  Moon,
  User,
  Banknote,
  DollarSign,
  IndianRupee,
  LogOut
} from 'lucide-react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import toast from 'react-hot-toast';

export default function Home() {
  const router = useRouter();
  const [darkMode, setDarkMode] = useState(false);
  const [hovered, setHovered] = useState(false);
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };
  const isLoggedIn = localStorage.getItem("token");
  return (
    <div className="min-h-screen bg-white dark:bg-black dark:text-white transition-colors duration-200">
      {/* Hero Section */}
      <header className="container mx-auto px-6 py-16">
        <nav className="flex justify-between items-center mb-16">
          <div className="flex items-center space-x-2">
            <LineChart className="w-8 h-8 text-primary dark:text-primary-light" />
            <span className="text-2xl font-bold text-primary-dark dark:text-white">FinOptima</span>
          </div>
          <div className="hidden md:flex space-x-8">
            <a href="#features" className="text-gray-800 dark:text-gray-200 hover:text-primary dark:hover:text-primary-light transition-colors"
              onClick={
                (e) => {
                  e.preventDefault();
                  document.getElementById("features")?.scrollIntoView({ behavior: "smooth" });
                }
              }>Features</a>
            <a href="#how-it-works" className="text-gray-800 dark:text-gray-200 hover:text-primary dark:hover:text-primary-light transition-colors"
              onClick={(e) => {
                e.preventDefault();
                document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" });
              }}>How it Works</a>

          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-primary dark:text-primary-light hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <button className="bg-primary hover:bg-primary-dark text-white px-6 py-2 rounded-lg transition-colors"
              onClick={() => {
                isLoggedIn?router.push("/dashboard"):router.push("/auth");
              }}
            >
              {isLoggedIn?"Profile":"LogIn"}
            </button>
            {
              isLoggedIn && <button className="flex items-center space-x-1 bg-gray-200 px-3 py-2 rounded-lg hover:bg-red-500 transition hover:text-white dark:bg-gray-900 dark:hover:bg-red-600" onClick={() => {
                localStorage.removeItem("token");
                toast.success("Logged Out Successfully");
                router.push("/");
              }}>
              <LogOut size={20} />
              <span>Logout</span>
            </button>
            }
          </div>
        </nav>

        <div className="flex flex-col md:flex-row items-center justify-between gap-12">
          <div className="md:w-1/2">
            <h1 className="text-5xl font-bold leading-tight mb-6 text-gray-900 dark:text-white">
              Smart Financial Planning for Your{' '}
              <span className="text-primary dark:text-primary-light">Better Future</span>
            </h1>
            <p className="text-lg text-gray-700 dark:text-gray-300 mb-8">
              Take control of your financial journey with AI-powered insights, risk detection,
              and personalized optimization strategies.
            </p>
            <div className="flex space-x-4 cursor-pointer">
              <motion.button
                onMouseEnter={() => setHovered(true)}
                onMouseLeave={() => setHovered(false)}
                onClick={() => {
                  isLoggedIn?router.push("/dashboard"):router.push("/auth");
                }}
                className={`relative overflow-hidden px-8 py-3 rounded-lg flex items-center space-x-2 transition-all duration-300 border-2 border-gray-200 text-shadow-lg ${hovered?"":"text-primary-dark"} dark:text-white hover:text-white`}
              >
                {/* Background Fill Animation */}
                <motion.span
                  initial={{ width: "0%" }}
                  animate={{ width: hovered ? "100%" : "0%" }}
                  transition={{ duration: 0.4, ease: "easeInOut" }}
                  className="absolute left-0 top-0 h-full bg-primary-dark"
                />

                {/* Text Content */}
                <span className="relative transition-all duration-300">
                  {hovered ? "Get Started Now" : "Ready ?"}
                </span>

                {/* Arrow Icon */}
                {hovered && <ArrowRight className="relative w-5 h-5 transition-transform duration-300" />}
                
              </motion.button>
            </div>
          </div>
          <div className="md:w-1/2">
            <IndianRupee size={512} className="text-primary dark:text-primary-light ml-auto mr-auto"/>
          </div>
        </div>
      </header>

      {/* Features Section */}
      <section className="py-20 bg-gray-50 dark:bg-gray-900" id="features">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-16 text-gray-900 dark:text-white">
            Comprehensive Financial Solutions
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: <Shield className="w-8 h-8 text-primary dark:text-primary-light" />,
                title: 'Risk Detection',
                description: 'Advanced algorithms to identify and prevent financial risks before they impact your portfolio.'
              },
              {
                icon: <BarChart3 className="w-8 h-8 text-primary dark:text-primary-light" />,
                title: 'Portfolio Optimization',
                description: 'Data-driven recommendations to maximize returns while maintaining your risk comfort level.'
              },
              {
                icon: <Brain className="w-8 h-8 text-primary dark:text-primary-light" />,
                title: 'Sentiment Analysis',
                description: 'Real-time market sentiment tracking to help you make informed investment decisions.'
              },
              {
                icon: <Target className="w-8 h-8 text-primary dark:text-primary-light" />,
                title: 'Goal Planning',
                description: 'Personalized strategies to eliminate debt and achieve your financial objectives.'
              },
              {
                icon: <GraduationCap className="w-8 h-8 text-primary dark:text-primary-light" />,
                title: 'Learning Path',
                description: 'Customized financial education journey tailored to your knowledge level and goals.'
              },
              {
                icon: <PiggyBank className="w-8 h-8 text-primary dark:text-primary-light" />,
                title: 'Savings Automation',
                description: 'Smart tools to automate your savings and accelerate wealth building.'
              }
            ].map((feature, index) => (
              <div key={index} className="bg-white dark:bg-black p-6 rounded-xl hover:shadow-lg transition-shadow">
                <div className="mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">{feature.title}</h3>
                <p className="text-gray-700 dark:text-gray-300">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
      <section className="py-20 bg-gray-50 dark:bg-gray-900" id="how-it-works">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4 text-gray-900 dark:text-white">How It Works</h2>
          <p className="text-lg text-gray-700 dark:text-gray-300">Follow these simple steps to manage your finances effortlessly.</p>
        </div>

        <div className="grid md:grid-cols-3 gap-12 px-6">
          {/* Step 1 */}
          <div className="relative flex flex-col items-center text-center bg-white p-8 rounded-xl shadow-lg hover:shadow-2xl transition-all border border-gray-200 dark:border-gray-700 dark:bg-black">
            <div className="bg-emerald-100 dark:bg-emerald-900 rounded-full p-4 mb-6">
              <User className="w-10 h-10 text-emerald-600 dark:text-emerald-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">Create Account</h3>
            <p className="text-gray-700 dark:text-gray-300">Sign up for free and set up your profile with essential financial details.</p>
            {/* Arrow */}
            <div className="hidden md:flex justify-center items-center absolute right-0 top-1/2 transform -translate-y-1/2">
              <ArrowRight className="w-10 h-10 text-emerald-500 dark:text-emerald-400 animate-pulse" />
            </div>
          </div>



          {/* Step 2 */}
          <div className="relative flex flex-col items-center text-center bg-white dark:bg-black p-8 rounded-xl shadow-lg hover:shadow-2xl transition-all border border-gray-200 dark:border-gray-700">
            <div className="bg-emerald-100 dark:bg-emerald-900 rounded-full p-4 mb-6">
              <Banknote className="w-10 h-10 text-emerald-600 dark:text-emerald-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">Connect Banks</h3>
            <p className="text-gray-700 dark:text-gray-300">Securely link your bank accounts to track transactions in real-time.</p>
            <div className="hidden md:flex justify-center items-center absolute right-0 top-1/2 transform -translate-y-1/2">
              <ArrowRight className="w-10 h-10 text-emerald-500 dark:text-emerald-400 animate-pulse" />
            </div>
          </div>

          {/* Arrow */}


          {/* Step 3 */}
          <div className="relative flex flex-col items-center text-center bg-white dark:bg-black p-8 rounded-xl shadow-lg hover:shadow-2xl transition-all border border-gray-200 dark:border-gray-700">
            <div className="bg-emerald-100 dark:bg-emerald-900 rounded-full p-4 mb-6">
              <BarChart3 className="w-10 h-10 text-emerald-600 dark:text-emerald-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">Track & Optimize</h3>
            <p className="text-gray-700 dark:text-gray-300">Get AI-powered insights to plan your budget and optimize savings.</p>
          </div>
        </div>
      </section>
      {/* Stats Section */}
      <section className="py-10 bg-primary dark:bg-primary-DARK">
        <div className="container mx-auto px-6">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold mb-2 text-white dark:text-primary-DARK">$2.5B+</div>
              <div className="text-primary-light">Assets Optimized</div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2 text-white">50,000+</div>
              <div className="text-primary-light">Active Users</div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2 text-white">98%</div>
              <div className="text-primary-light">Success Rate</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold mb-8 text-gray-900 dark:text-white">Ready to Optimize Your Finances?</h2>
          <div className="flex flex-col md:flex-row justify-center items-center space-y-4 md:space-y-0 md:space-x-6">
            <button className="bg-primary hover:bg-primary-dark text-white px-8 py-3 rounded-lg flex items-center justify-center space-x-2 transition-colors w-full md:w-auto"
              onClick={() => {
                isLoggedIn?router.push("/dashboard"):router.push("/auth");
              }
              }
            >
              <span>Get Started Now</span>
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-100 py-12 dark:bg-black">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-2 mb-4 md:mb-0">
              <LineChart className="w-6 h-6 text-primary" />
              <span className="text-xl font-bold text-black dark:text-white">FinOptima</span>
            </div>
            <div className="text-black dark:text-gray-400">
              © 2025 FinOptima. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
