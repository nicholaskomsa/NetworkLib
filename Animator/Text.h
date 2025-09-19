#pragma once

#include <GL/glew.h>
#include <freetype/freetype.h>

#include <string>
#include <vector>

class TextManager {

public:
	FT_Library mFreeType;
	FT_Face mFace;

	void create(const std::string& fontName, FT_UInt fontSize){

		FT_Init_FreeType(&mFreeType);

		FT_New_Face(mFreeType, fontName.c_str(), 0, &mFace);
		FT_Set_Pixel_Sizes(mFace, 0, fontSize);
	}

	void destroy() {

		FT_Done_Face(mFace);       // Destroys the font face
		FT_Done_FreeType(mFreeType);     // Shuts down the FreeType library
	}
};

class Text {

	GLuint mTexture = 0;
	FT_Int mTotalWidth = 0, mTotalHeight = 0, mMaxAscent = 0, mMaxDescent = 0;
	std::vector<std::uint32_t> mTextBuffer;

	void create(const std::string& text, std::uint32_t color, const FT_Face& face) {

		for (auto c: text ){

			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;

			mTotalWidth += g->advance.x >> 6; // advance in pixels
			mMaxAscent = std::max(mMaxAscent, g->bitmap_top);
			mMaxDescent = std::max(mMaxDescent, FT_Int(g->bitmap.rows - g->bitmap_top));
		}
		mTotalHeight = mMaxAscent + mMaxDescent;

		mTextBuffer.resize(mTotalWidth * mTotalHeight, 0);
		
		int penX = 0;
		for (char c : text) {

			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;
			FT_Bitmap& bmp = g->bitmap;

			int xOffset = penX + g->bitmap_left;
			int yOffset = mMaxAscent - g->bitmap_top;

			std::uint8_t* userColor = reinterpret_cast<std::uint8_t*>(&color);

			for (int row = 0; row < bmp.rows; ++row) {
				for (int col = 0; col < bmp.width; ++col) {
					int x = xOffset + col;
					int y = yOffset + row;

					float greyScale = bmp.buffer[row * bmp.pitch + col] / 255.0f;
					std::uint8_t* savedColor = reinterpret_cast<std::uint8_t*>(&mTextBuffer[y * mTotalWidth + x]);

					savedColor[0] = greyScale * userColor[0]; // R
					savedColor[1] = greyScale * userColor[1]; // G
					savedColor[2] = greyScale * userColor[2]; // B
					savedColor[3] = 1.0 * userColor[3]; // A
				}
			}

			penX += g->advance.x >> 6;
		}

		glGenTextures(1, &mTexture);
		glBindTexture(GL_TEXTURE_2D, mTexture);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
			mTotalWidth, mTotalHeight
			, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, mTextBuffer.data());

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void destroy(){
		if (mTexture) {
			glDeleteTextures(1, &mTexture);
			mTexture = 0;
		}
		mTextBuffer.clear();
		mTextBuffer.shrink_to_fit();
	}

};