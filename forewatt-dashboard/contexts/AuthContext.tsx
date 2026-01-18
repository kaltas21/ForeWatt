import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// User types
export type UserRole = 'admin' | 'client';

export interface User {
  username: string;
  role: UserRole;
  displayName: string;
  email?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isAdmin: boolean;
  login: (username: string, password: string) => Promise<{ success: boolean; error?: string }>;
  signup: (username: string, password: string, email: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
}

// Hardcoded users - In production, use Firebase Auth or similar
const USERS: Record<string, { password: string; role: UserRole; displayName: string; email?: string }> = {
  admin1: { password: 'ForeWatt@2024', role: 'admin', displayName: 'Admin User 1', email: 'admin1@forewatt.com' },
  admin2: { password: 'ForeWatt@2024', role: 'admin', displayName: 'Admin User 2', email: 'admin2@forewatt.com' },
  admin3: { password: 'ForeWatt@2024', role: 'admin', displayName: 'Admin User 3', email: 'admin3@forewatt.com' },
  client1: { password: 'Client@123', role: 'client', displayName: 'Demo Client', email: 'client1@example.com' },
};

// Storage for dynamically registered users (persisted in localStorage)
const STORAGE_KEY = 'forewatt_registered_users';

const getRegisteredUsers = (): Record<string, { password: string; role: UserRole; displayName: string; email?: string }> => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
};

const saveRegisteredUsers = (users: Record<string, { password: string; role: UserRole; displayName: string; email?: string }>) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(users));
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);

  // Check for existing session on mount
  useEffect(() => {
    const storedUser = localStorage.getItem('forewatt_user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch {
        localStorage.removeItem('forewatt_user');
      }
    }
  }, []);

  const login = async (username: string, password: string): Promise<{ success: boolean; error?: string }> => {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500));

    const normalizedUsername = username.toLowerCase().trim();

    // Check hardcoded users first, then registered users
    const allUsers = { ...USERS, ...getRegisteredUsers() };
    const userData = allUsers[normalizedUsername];

    if (!userData) {
      return { success: false, error: 'User not found. Please sign up first.' };
    }

    if (userData.password !== password) {
      return { success: false, error: 'Invalid password. Please try again.' };
    }

    const loggedInUser: User = {
      username: normalizedUsername,
      role: userData.role,
      displayName: userData.displayName,
      email: userData.email,
    };

    setUser(loggedInUser);
    localStorage.setItem('forewatt_user', JSON.stringify(loggedInUser));

    return { success: true };
  };

  const signup = async (username: string, password: string, email: string): Promise<{ success: boolean; error?: string }> => {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500));

    const normalizedUsername = username.toLowerCase().trim();

    // Check if username already exists
    const allUsers = { ...USERS, ...getRegisteredUsers() };
    if (allUsers[normalizedUsername]) {
      return { success: false, error: 'Username already exists. Please choose another.' };
    }

    // Validate password strength
    if (password.length < 6) {
      return { success: false, error: 'Password must be at least 6 characters.' };
    }

    // Validate email
    if (!email.includes('@')) {
      return { success: false, error: 'Please enter a valid email address.' };
    }

    // Register new user as client
    const registeredUsers = getRegisteredUsers();
    registeredUsers[normalizedUsername] = {
      password,
      role: 'client',
      displayName: username,
      email,
    };
    saveRegisteredUsers(registeredUsers);

    // Auto-login after signup
    const newUser: User = {
      username: normalizedUsername,
      role: 'client',
      displayName: username,
      email,
    };

    setUser(newUser);
    localStorage.setItem('forewatt_user', JSON.stringify(newUser));

    return { success: true };
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('forewatt_user');
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isAdmin: user?.role === 'admin',
        login,
        signup,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
