import axios from "axios";

const API_URL = "http://localhost:5003/api/auth"; // Replace with actual backend URL

export const loginUser = async (email:any, password:any) => {
  try {
    const response = await axios.post(`${API_URL}/login`, { email, password });
    return response.data;
  } catch (error:any) {
    throw error.response?.data?.message || "Login failed";
  }
};

export const registerUser = async (fullName:any, email:any, password:any) => {
  try {
    const response = await axios.post(`${API_URL}/signup`, {
      fullName,
      email,
      password,
    });
    return response.data;
  } catch (error:any) {
    throw error.response?.data?.message || "Signup failed";
  }
};
